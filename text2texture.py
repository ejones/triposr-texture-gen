import argparse
import copy
import io
import os
import os.path
import shlex
import subprocess
import sys

import numpy as np
import open3d as o3d
import scipy.interpolate
import scipy.spatial
from PIL import Image


def process_tripo_mesh(mesh):
    rot = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, -np.pi / 2))
    new_mesh = copy.deepcopy(mesh)
    new_mesh.rotate(rot)
    new_mesh.remove_non_manifold_edges()
    new_mesh = new_mesh.simplify_quadric_decimation(10000)
    return new_mesh


def raycast_mesh(tmesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    rays = scene.create_rays_pinhole(fov_deg=60,
                                     center=[0, 0, 0],
                                     eye=[0, 0, 1.3],
                                     up=[0, -1, 0],
                                     width_px=512,
                                     height_px=512)

    return scene.cast_rays(rays)


def ray_hits_to_depth(raycast_result):
    hits = raycast_result['t_hit'].numpy()
    hits[hits == np.inf] = 0 # REVIEW: can skip for later?

    min1 = np.unique(hits)[1] # min except for 0, mayber better way?
    hits2 = (((np.max(hits) - hits) / (np.max(hits) - min1)).clip(0, 1) * 255).astype('u1')
    hits2[hits == 0] = 0

    return Image.fromarray(hits2)


# - ans_uvs : [N, 2] array
# - ans_prim_ids : [N] array
# - point_colors : [N, 3]
def compute_texture(tmesh, ans_uvs, ans_prim_ids, point_colors, size=512, imdata=None):
    if imdata is None:
        imdata = np.ones((size, size, 3), 'u1') * 255

    # prepend 1 - uv1 - uv2 to make [N, 3] array
    ans_uvs_3 = np.insert(ans_uvs, 0, (1 - np.sum(ans_uvs, 1)), axis=1)

    # Index per-triangle vertex x UVs on triangle IDs from fit to get [N, 3, 2] array
    triuvs = tmesh.triangle.texture_uvs.numpy()[ans_prim_ids]

    # Dot each UV with each triangle UV -> [N, 2] array
    uvs = np.einsum('ij,ijk->ik', ans_uvs_3, triuvs)

    imxy = (uvs * size).astype('u2') # assume size <= max(uint16)

    # interpolate missing pixels...
    interp = scipy.interpolate.LinearNDInterpolator(imxy, point_colors.astype('f4') / 255, 1)
    all_xs_ys = np.indices((size, size)).reshape(2, -1) # [2, N] array
    all_points = all_xs_ys.transpose(1, 0) # [N, 2]

    # ...and mask out points too far from reference points
    kdtree = scipy.spatial.KDTree(imxy)
    dists = kdtree.query(all_points)[0]

    xs, ys = all_xs_ys[:, dists < 2]

    colors = interp(xs, ys)
    imdata[(size - 1) - ys, xs] = colors * 255
    #imdata[(size - 1) - imxy[:, 1], imxy[:, 0]] = point_colors

    return imdata


def compute_raycast_texture(tmesh, raycast_result, rgb_im, size=512):
    print()
    print('computing UV atlas for', len(tmesh.triangle.indices), 'triangles')
    tmesh.compute_uvatlas(size, parallel_partitions=2)
    
    print()
    print('generating texture')
    imdata = tmesh.bake_vertex_attr_textures(size, {'colors'})['colors'].numpy()

    imdata = (imdata * 255).astype('u1')
    prim_ids = raycast_result['primitive_ids'].numpy().flatten()
    mask = prim_ids != 0xffff_ffff
    return compute_texture(
        tmesh,
        raycast_result['primitive_uvs'].numpy().reshape(-1, 2)[mask],
        prim_ids[mask],
        np.array(rgb_im).reshape(-1, 3)[mask],
        size,
        imdata,
    )


def set_tmesh_tex(tmesh, tex_imdata):
    tmesh.material.set_default_properties()
    tmesh.material.material_name = 'defaultLit'
    tmesh.material.texture_maps['albedo'] = o3d.t.geometry.Image(tex_imdata)
    if 'colors' in tmesh.vertex:
        del tmesh.vertex['colors']


def text2texture(mesh, desc, steps, depth_txt2img_path, img_model, device, out_path_base):
    print()
    print('processing mesh')
    mesh = process_tripo_mesh(mesh)

    # print(f'saving preprocessed mesh at {out_path_base}-preproc.obj')
    # o3d.io.write_triangle_mesh(f'{out_path_base}-preproc.obj', mesh)

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    raycast_result = raycast_mesh(tmesh)
    depth_im = ray_hits_to_depth(raycast_result)
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
        tex_imdata = compute_raycast_texture(tmesh, raycast_result, Image.open(painted_path))
        proc.kill()

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
