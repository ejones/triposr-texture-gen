import argparse
import copy
import collections
import math
import io
import os
import os.path
import shlex
import subprocess
import sys

import numpy as np
import open3d as o3d
import scipy
from PIL import Image


def process_tripo_mesh(mesh):
    rot = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, -np.pi / 2))
    new_mesh = copy.deepcopy(mesh)
    new_mesh.rotate(rot)
    new_mesh.remove_non_manifold_edges()
    new_mesh = new_mesh.simplify_quadric_decimation(10000)
    return new_mesh


def raycast_mesh(tmesh, camera_dist=2.8, rot_x_rad=0.0, rot_y_rad=0.0, fov_deg=30, size=512):
    if rot_x_rad or rot_y_rad:
        rot = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz((rot_x_rad, rot_y_rad, 0.0))
        new_mesh = tmesh.clone()
        new_mesh.rotate(rot, new_mesh.get_center())
    else:
        new_mesh = tmesh

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(new_mesh)

    rays = scene.create_rays_pinhole(fov_deg=fov_deg,
                                     center=[0, 0, 0],
                                     eye=[0, 0, camera_dist],
                                     up=[0, -1, 0],
                                     width_px=size,
                                     height_px=size)

    result = scene.cast_rays(rays)

    np_result = {
        k: result[k].numpy()
        for k in ('primitive_ids', 'primitive_uvs', 'primitive_normals')
    }

    hits = result['t_hit'].numpy()
    hits[hits == np.inf] = 0 # REVIEW: can skip for later?

    min1 = np.unique(hits)[1] # min except for 0, mayber better way?
    hits2 = (((np.max(hits) - hits) / (np.max(hits) - min1)).clip(0, 1) * 255).astype('u1')
    hits2[hits == 0] = 0

    np_result['depth'] = hits2

    return np_result


def find_raycast_bounding_box(raycast_result):
    _, slicex = scipy.ndimage.find_objects(raycast_result['primitive_ids'] != 0xffff_ffff)[0]
    slicex = slice(max(slicex.start - 20, 0), slicex.stop + 20)

    # Stable Diffusion dims must be divisible by 8
    slicex = slice(slicex.start, slicex.stop - ((slicex.stop - slicex.start) % 8))

    return slice(0, raycast_result['primitive_ids'].shape[0]), slicex


def raycast_mesh_multi(
    tmesh, camera_dist=2.8, xy_angles_rad=[(0, 0), (0, np.pi)], crop=False, size=512,
):
    raycasts = [raycast_mesh(tmesh, 2.8, rx, -ry, size=size) for rx, ry in xy_angles_rad]
    boxes = [find_raycast_bounding_box(r) for r in raycasts]

    merged_raycast = {
        key: np.concatenate(
            [
                (r[key][:, sx] if crop else r[key])
                for r, (_, sx) in zip(raycasts, boxes)
            ],
            axis=1,
        )
        for key in ('depth', 'primitive_ids', 'primitive_uvs', 'primitive_normals')
    }

    merged_boxes = []
    start = 0
    for sy, sx in boxes:
        stop = start + sx.stop - sx.start
        merged_boxes.append((sy, slice(start, stop)))
        start = stop

    return merged_raycast, merged_boxes


def generate_normal_map(raycast):
    return Image.fromarray(((raycast['primitive_normals'] + 1) * 127.5).astype('u1'))


def interpolate_pixels(imdata, missing_mask, known_mask=None):
    known_yx = np.argwhere(~missing_mask if known_mask is None else (~missing_mask & known_mask))
    interp = scipy.interpolate.LinearNDInterpolator(
        known_yx, imdata[known_yx[:, 0], known_yx[:, 1]].astype('f4') / 255, 1,
    )
    new_imdata = np.array(imdata)
    missing_ys_xs = np.nonzero(missing_mask)
    new_imdata[missing_ys_xs] = interp(*missing_ys_xs) * 255
    return new_imdata


# - ans_uvs : [N, 2] array
# - ans_prim_ids : [N] array
# - depth: [N]
# - point_colors : [N, 3]
def compute_texture(tmesh, ans_uvs, ans_prim_ids, depth, point_colors, size=512):
    imdata = np.zeros((size, size, 4), 'u1')

    # prepend 1 - uv1 - uv2 to make [N, 3] array
    ans_uvs_3 = np.insert(ans_uvs, 0, (1 - np.sum(ans_uvs, 1)), axis=1)

    # Index per-triangle vertex x UVs on triangle IDs from fit to get [N, 3, 2] array
    triuvs = tmesh.triangle.texture_uvs.numpy()[ans_prim_ids]

    # Dot each UV with each triangle UV -> [N, 2] array
    uvs = np.einsum('ij,ijk->ik', ans_uvs_3, triuvs)

    imxy = (uvs * size).astype('u2') # assume size <= max(uint16)

    # interpolate missing pixels...
    interp = scipy.interpolate.LinearNDInterpolator(
        imxy,
        np.concatenate((point_colors, depth[:, None]), axis=-1).astype('f4') / 255,
        1,
    )
    all_xs_ys = np.indices((size, size)).reshape(2, -1) # [2, N] array
    all_points = all_xs_ys.transpose(1, 0) # [N, 2]

    # ...and mask out points too far from reference points
    kdtree = scipy.spatial.KDTree(imxy)
    dists = kdtree.query(all_points)[0]

    xs, ys = all_xs_ys[:, dists < 2]

    colors = interp(xs, ys)
    imdata[(size - 1) - ys, xs] = colors * 255

    return imdata


def compute_raycast_texture(
    tmesh, raycast_result, rgb_imdata, slice_sets, tex_imdata, size=512, mask=None, erode=False,
):
    print()
    print('generating texture')

    non_inf_rays = raycast_result['primitive_ids'] != 0xffff_ffff
    mask = (mask & non_inf_rays) if mask is not None else non_inf_rays
    if erode:
        mask = scipy.ndimage.binary_erosion(mask, np.ones((4, 4)))

    def get_slices(arr, slices):
        # TODO assumes consistent y-size
        return np.concatenate(
            [arr[sy, sx] for sy, sx in slices],
            axis=1,
        )

    layers = []
    for slices in slice_sets:
        slice_mask = get_slices(mask, slices).flatten()

        uvs = get_slices(raycast_result['primitive_uvs'], slices).reshape(-1, 2)[slice_mask]
        ids = get_slices(raycast_result['primitive_ids'], slices).flatten()[slice_mask]
        rgb = get_slices(rgb_imdata, slices).reshape(-1, 3)[slice_mask]
        depth = get_slices(raycast_result['depth'], slices).flatten()[slice_mask]

        layers.append(compute_texture(tmesh, uvs, ids, depth, rgb, size))

    # alpha-weighted average of texture layers (alpha is derived from depth)
    blended = np.average(
        [layer[..., :3] for layer in layers],
        axis=0,
        weights=[
            np.clip(layer[..., [3]].repeat(3, axis=-1).astype('f4') ** 4, 1, None)
            for layer in layers
        ],
    ).astype('u1')

    missing_tex = np.all([layer[..., 3] == 0 for layer in layers], axis=0)
    blended[missing_tex] = tex_imdata[missing_tex]

    return blended


def set_tmesh_tex(tmesh, tex_imdata):
    tmesh.material.set_default_properties()
    tmesh.material.material_name = 'defaultUnlit'
    tmesh.material.texture_maps['albedo'] = o3d.t.geometry.Image(tex_imdata)
    if 'colors' in tmesh.vertex:
        del tmesh.vertex['colors']


def text2texture(
        mesh,
        desc,
        steps,
        depth_txt2img_path,
        img_model,
        device,
        out_path_base,
        size=512,
    ):
    print()
    print('processing mesh')
    mesh = process_tripo_mesh(mesh)

    # print(f'saving preprocessed mesh at {out_path_base}-preproc.obj')
    # o3d.io.write_triangle_mesh(f'{out_path_base}-preproc.obj', mesh)

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    print()
    print('computing UV atlas for', len(tmesh.triangle.indices), 'triangles')
    tmesh.compute_uvatlas(size, parallel_partitions=2)

    missing_tex_rgb = [0, 255, 0]
    tex_imdata = np.full((size, size, 3), missing_tex_rgb, 'u1')

    raycast, raycast_slices = raycast_mesh_multi(
        tmesh, 2.8, [(0, i * math.pi / 2) for i in range(4)], crop=True, size=size,
    )
    depth_im = generate_normal_map(raycast)
    depth_path = f'{out_path_base}-preproc-depth.png'

    print('saving depth map at', depth_path)
    depth_im.save(depth_path)

    painted_path = f'{out_path_base}-preproc-depth-paint.png'
    depth_paint_args = [
        depth_txt2img_path,
        desc, 
        depth_path,
        painted_path,
        '--steps',
        str(steps),
        '--image-model',
        img_model,
        *(['--device', device] if device else [])
    ]

    print()
    print('>', *(shlex.quote(arg) for arg in depth_paint_args))
    subprocess.run(
        [sys.executable, *depth_paint_args],
        check=True,
        env={'PYTORCH_ENABLE_MPS_FALLBACK': '1'},
    )

    with subprocess.Popen(['bash', '-c', 'while true; do echo -n .; sleep 0.5; done']) as proc:
        tex_imdata = compute_raycast_texture(
            tmesh,
            raycast,
            np.array(Image.open(painted_path)),
            [[raycast_slices[0], raycast_slices[2]], [raycast_slices[1], raycast_slices[3]]],
            tex_imdata,
            size,
            erode=True,
        )
        proc.kill()

    # Interpolate any remaining missing texture regions as a fallback
    tex_imdata = interpolate_pixels(tex_imdata, np.all(tex_imdata == missing_tex_rgb, axis=-1))

    set_tmesh_tex(tmesh, tex_imdata)
    return tmesh


def write_mesh(out_base, tmesh):
    out_mesh_path = f'{out_base}.obj'
    o3d.t.io.write_triangle_mesh(out_mesh_path, tmesh)

    # Open 3D seems to have spotty support for writing textures, so manually
    # write out the texture images + update MTL file to reference them
    o3d.t.io.write_image(f'{out_base}.png', tmesh.material.texture_maps['albedo'])
    map_ref_path = os.path.basename(f'{out_base}.png')

    with open(f'{out_base}.mtl', 'a') as mtl_file:
        mtl_file.write(f'\nmap_Ka {map_ref_path}\nmap_Kd {map_ref_path}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mesh', help='Path to input 3D model file (e.g. mesh.obj)')
    parser.add_argument('desc', help='Short description of desired model appearance')
    parser.add_argument(
        '--image-model',
        help='SD 1.5-based model for texture image gen',
        default='Lykon/dreamshaper-8',
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=12,
        help='Num inference steps for texture image gen',
    )
    parser.add_argument(
        '--device',
        default='',
        type=str,
        help='Device to prefer. Default: try to auto-detect from platform (CUDA, Metal)'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory to save the results',
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    out_base = os.path.join(args.output_dir, 'mesh')

    tmesh = text2texture(
        mesh=mesh,
        desc=args.desc,
        steps=args.steps,
        depth_txt2img_path=os.path.join(os.path.dirname(__file__), 'depth_txt2img.py'),
        img_model=args.image_model,
        device=args.device,
        out_path_base=out_base,
    )

    out_mesh_base = f'{out_base}-tex'
    print('writing new mesh to', f'{out_mesh_base}.obj')
    write_mesh(out_mesh_base, tmesh)

    if sys.stdin.isatty():
        o3d.visualization.draw(tmesh)
