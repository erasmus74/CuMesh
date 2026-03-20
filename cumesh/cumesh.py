from typing import *
import math
import torch
from tqdm import tqdm
from .xatlas import Atlas
from . import _C


class CuMesh:
    def __init__(self):
        self.cu_mesh = _C.CuMesh()

    def init(self, vertices: torch.Tensor, faces: torch.Tensor):
        """
        Initialize the CuMesh with vertices and faces.

        Args:
            vertices: a tensor of shape [V, 3] containing the vertex positions.
            faces: a tensor of shape [F, 3] containing the face indices.
        """
        assert vertices.ndim == 2 and vertices.shape[1] == 3, "Input vertices must be of shape [V, 3]"
        assert faces.ndim == 2 and faces.shape[1] == 3, "Input faces must be of shape [F, 3]"
        assert vertices.is_contiguous() and faces.is_contiguous(), "Input tensors must be contiguous"
        assert vertices.is_cuda and faces.is_cuda and vertices.device == faces.device, "Input tensors must both be on the same CUDA device"
        self.cu_mesh.init(vertices, faces)
        
    @property
    def num_vertices(self) -> int:
        return self.cu_mesh.num_vertices()
    
    @property
    def num_faces(self) -> int:
        return self.cu_mesh.num_faces()
    
    @property
    def num_edges(self) -> int:
        return self.cu_mesh.num_edges()
    
    @property
    def num_boundaries(self) -> int:
        return self.cu_mesh.num_boundaries()
    
    @property
    def num_conneted_components(self) -> int:
        return self.cu_mesh.num_conneted_components()
    
    @property
    def num_boundary_conneted_components(self) -> int:
        return self.cu_mesh.num_boundary_conneted_components()
    
    @property
    def num_boundary_loops(self) -> int:
        return self.cu_mesh.num_boundary_loops()

    def clear_cache(self):
        """
        Clear the cached data.
        """
        self.cu_mesh.clear_cache()

    def read(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read the current vertices and faces from the CuMesh.

        Returns:
            A tuple of two tensors: the vertex positions and the face indices.
        """
        return self.cu_mesh.read()
    
    def read_face_normals(self) -> torch.Tensor:
        """
        Read the normals of the faces from the CuMesh.

        Returns:
            The face normals as an [F, 3] tensor.
        """
        return self.cu_mesh.read_face_normals()
    
    def read_vertex_normals(self) -> torch.Tensor:
        """
        Read the normals of the vertices from the CuMesh.

        Returns:
            The vertex normals as an [V, 3] tensor.
        """
        return self.cu_mesh.read_vertex_normals()
    
    def read_edges(self) -> torch.Tensor:
        """
        Read the edges of the mesh from the CuMesh.

        Returns:
            A tensor of shape [E, 2] containing the edge indices.
        """
        return self.cu_mesh.read_edges()
    
    def read_boundaries(self) -> torch.Tensor:
        """
        Read the boundary edges of the mesh from the CuMesh.

        Returns:
            A tensor of shape [B] containing the boundary edge indices.
        """
        return self.cu_mesh.read_boundaries()
    
    
    def read_manifold_face_adjacency(self) -> torch.Tensor:
        """
        Read the manifold face adjacency from the CuMesh.

        Returns:
            A tensor of shape [M, 2] containing the manifold face adjacency.
        """
        return self.cu_mesh.read_manifold_face_adjacency()
    
    def read_manifold_boundary_adjacency(self) -> torch.Tensor:
        """
        Read the manifold boundary adjacency from the CuMesh.

        Returns:
            A tensor of shape [M, 2] containing the manifold boundary adjacency.
        """
        return self.cu_mesh.read_manifold_boundary_adjacency()
    
    def read_connected_components(self) -> Tuple[int, torch.Tensor]:
        """
        Read the connected component IDs for each face.

        Returns:
            A tuple of two values:
                - the number of connected components
                - a tensor of shape [F] containing the connected component ID for each face.
        """
        return self.cu_mesh.read_connected_components()
    
    def read_boundary_connected_components(self) -> Tuple[int, torch.Tensor]:
        """
        Read the connected component IDs for each boundary edge.

        Returns:
            A tuple of two values:
                - the number of connected components
                - a tensor of shape [E] containing the connected component ID for each boundary edge.
        """
        return self.cu_mesh.read_boundary_connected_components()
    
    def read_boundary_loops(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Read the boundary loops of the mesh.

        Returns:
            A tuple of three values:
                - the number of boundary loops
                - a tensor of shape [L] containing the indices of the boundary edges in each loop.
                - a tensor of shape [N_loops + 1] containing the offsets of the boundary edges in each loop.
        """
        return self.cu_mesh.read_boundary_loops()
    
    def read_all_cache(self) -> Dict[str, torch.Tensor]:
        """
        Read all cached data.

        Returns:
            A dictionary of cached data.
        """
        return self.cu_mesh.read_all_cache()
    
    def compute_face_normals(self):
        """
        Compute the normals of the faces.
        """
        self.cu_mesh.compute_face_normals()
    
    def compute_vertex_normals(self):
        """
        Compute the normals of the vertices.
        """
        self.cu_mesh.compute_vertex_normals()
        
    def get_vertex_face_adjacency(self):
        """
        Compute the vertex to face adjacency.
        """
        self.cu_mesh.get_vertex_face_adjacency()
        
    def get_edges(self):
        """
        Compute the edges of the mesh.
        """
        self.cu_mesh.get_edges()
        
    def get_edge_face_adjacency(self):
        """
        Compute the edge to face adjacency.
        """
        self.cu_mesh.get_edge_face_adjacency()
        
    def get_vertex_edge_adjacency(self):
        """
        Compute the vertex to edge adjacency.
        """
        self.cu_mesh.get_vertex_edge_adjacency()
        
    def get_boundary_info(self):
        """
        Compute the boundary information of the mesh.
        """
        self.cu_mesh.get_boundary_info()
        
    def get_vertex_boundary_adjacency(self):
        """
        Compute the vertex to boundary adjacency.
        """
        self.cu_mesh.get_vertex_boundary_adjacency()
        
    def get_manifold_face_adjacency(self):
        """
        Compute the manifold face adjacency.
        """
        self.cu_mesh.get_manifold_face_adjacency()
        
    def get_manifold_boundary_adjacency(self):
        """
        Compute the manifold boundary adjacency.
        """
        self.cu_mesh.get_manifold_boundary_adjacency()
        
    def get_connected_components(self):
        """
        Compute the connected components of the mesh.
        """
        self.cu_mesh.get_connected_components()
        
    def get_boundary_connected_components(self):
        """
        Compute the connected components of the boundary of the mesh.
        """
        self.cu_mesh.get_boundary_connected_components()
        
    def get_boundary_loops(self):
        """
        Compute the boundary loops of the mesh.
        """
        self.cu_mesh.get_boundary_loops()
        
    def remove_faces(self, face_mask: torch.Tensor):
        """
        Remove faces from the mesh.

        Args:
            face_mask: a boolean tensor of shape [F] indicating which faces to remove.
        """
        assert face_mask.ndim == 1 and face_mask.shape[0] == self.num_faces, "face_mask must be a boolean tensor of shape [F]"
        assert face_mask.is_contiguous() and face_mask.is_cuda, "face_mask must be a CUDA tensor"
        assert face_mask.dtype == torch.bool, "face_mask must be a boolean tensor"
        self.cu_mesh.remove_faces(face_mask)
    
    def remove_unreferenced_vertices(self):
        """
        Remove unreferenced vertices from the mesh.
        """
        self.cu_mesh.remove_unreferenced_vertices()
        
    def remove_duplicate_faces(self):
        """
        Remove duplicate faces from the mesh.
        """
        self.cu_mesh.remove_duplicate_faces()
        
    def remove_degenerate_faces(self):
        """
        Remove degenerate faces from the mesh.
        """
        self.cu_mesh.compute_face_normals()
        face_normals = self.cu_mesh.read_face_normals()
        kept = (face_normals.isnan().sum(dim=1) == 0)
        self.remove_faces(kept)
        
    def fill_holes(self, max_hole_perimeter: float=3e-2):
        """
        Fill holes in the mesh.

        Args:
            max_hole_perimeter: the maximum perimeter of a hole to fill.
        """
        self.cu_mesh.fill_holes(max_hole_perimeter)
        
    def repair_non_manifold_edges(self):
        """
        Repair Non-manifold edges by splitting vertices.
        This creates duplicate vertices with the same coordinates.
        """
        self.cu_mesh.repair_non_manifold_edges()

    def remove_non_manifold_faces(self):
        """
        Remove faces on non-manifold edges.
        For each non-manifold edge (shared by >2 faces), only keep the first 2 faces.
        This repairs non-manifold edges by deleting faces instead of splitting vertices.
        """
        self.cu_mesh.remove_non_manifold_faces()
        
    def remove_small_connected_components(self, min_area: float):
        """
        Repair Non-manifold edges by splitting edges
        
        Args:
            min_area: the minimum area of a connected component to keep.
        """
        self.cu_mesh.remove_small_connected_components(min_area)
        
    def unify_face_orientations(self):
        """
        Unify the orientations of the faces.
        """
        self.cu_mesh.unify_face_orientations()
    
    def simplify(self, target_num_faces: int, verbose: bool=False, options: dict={}):
        """
        Simplifies the mesh using a fast approximation algorithm with gpu acceleration.

        Args:
            target_num_faces: the target number of faces to simplify to.
            verbose: whether to print the progress of the simplification.
            options: a dictionary of options for the simplification algorithm.
        """
        assert isinstance(target_num_faces, int) and target_num_faces > 0, "target_num_faces must be a positive integer"

        num_face = self.cu_mesh.num_faces()
        if num_face <= target_num_faces:
            return
        
        if verbose:
            pbar = tqdm(total=num_face-target_num_faces, desc="Simplifying", disable=not verbose)

        thresh = options.get('thresh', 1e-8)
        lambda_edge_length = options.get('lambda_edge_length', 1e-2)
        lambda_skinny = options.get('lambda_skinny', 1e-3)
        while True:
            if verbose:
                pbar.set_description(f"Simplifying [thres={thresh:.2e}]")
            
            new_num_vert, new_num_face = self.cu_mesh.simplify_step(lambda_edge_length, lambda_skinny, thresh, False)
            
            if verbose:
                pbar.update(num_face - max(target_num_faces, new_num_face))

            if new_num_face <= target_num_faces:
                break
            
            del_num_face = num_face - new_num_face
            if del_num_face / num_face < 1e-2:
                thresh *= 10
            num_face = new_num_face
            
        if verbose:
            pbar.close()
            
    def compute_charts(
        self,
        threshold_cone_half_angle_rad: float=math.radians(90),
        refine_iterations: int=100,
        global_iterations: int=3,
        smooth_strength: float=1,
        area_penalty_weight: float=0.1,
        perimeter_area_ratio_weight: float=0.0001,
    ):
        """
        Compute the atlas charts.

        Args:
            threshold_cone_half_angle_rad: The threshold for the cone half angle in radians.
            refine_iterations: The number of refinement iterations.
            smooth_strength: The strength of chart boundary smoothing.
            area_penalty_weight: Coefficient for chart size penalty. Cost += Area * weight.
                                 Prevents charts from becoming too large if > 0, 
                                 or encourages larger charts if < 0 (though usually used to penalize size variance).
            perimeter_area_ratio_weight: Coefficient for shape irregularity (long-strip) penalty. 
                                         Cost += (Perimeter / Area) * weight.
                                         Higher values penalize long strips and encourage circular/compact shapes.
        """
        self.cu_mesh.compute_charts(
            threshold_cone_half_angle_rad,
            refine_iterations,
            global_iterations,
            smooth_strength,
            area_penalty_weight,
            perimeter_area_ratio_weight
        )
        
    def read_atlas_charts(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Read the atlas chart IDs for each face.
        
        Returns:
            A tuple of two values:
                - the number of charts
                - a tensor of shape [F] containing the chart ID for each face.
                - a tensor of shape [V] containing the vertex map
                - a tensor of shape [F, 3] containing the chart faces
                - a tensor of shape [C+1] containing the offsets of the chart vertices in the vertices tensor.
                - a tensor of shape [C+1] containing the offsets of the chart faces in the faces tensor.
        """
        return self.cu_mesh.read_atlas_charts()
    
    @staticmethod
    def _gpu_uv_parameterize(new_vertices, num_charts, chart_vmap, chart_faces,
                              chart_vertex_offset, chart_face_offset,
                              padding_pixels=1, resolution=1024, verbose=False):
        """
        GPU-based UV parameterization using PCA projection and shelf packing.
        Replaces xatlas for ROCm/HIP where xatlas (CPU) is too slow.
        """
        device = new_vertices.device
        chart_vertices = new_vertices[chart_vmap]
        total_verts = chart_vmap.shape[0]

        all_uvs = torch.zeros((total_verts, 2), dtype=torch.float32, device=device)
        # Track each chart's UV bounding box size for packing
        chart_widths = []
        chart_heights = []

        for i in tqdm(range(num_charts), desc="GPU UV parameterize", disable=not verbose):
            v_start = chart_vertex_offset[i].item()
            v_end = chart_vertex_offset[i + 1].item()
            n_verts = v_end - v_start

            if n_verts < 3:
                chart_widths.append(1e-6)
                chart_heights.append(1e-6)
                continue

            verts = chart_vertices[v_start:v_end]

            # PCA: project onto the two directions of greatest variance
            center = verts.mean(dim=0)
            centered = verts - center
            cov = centered.T @ centered
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # eigh returns ascending order; last two are largest
            axis1 = eigenvectors[:, -1]
            axis2 = eigenvectors[:, -2]

            u = centered @ axis1
            v = centered @ axis2
            uvs = torch.stack([u, v], dim=1)

            # Normalize to [0, 1]
            uv_min = uvs.min(dim=0).values
            uv_max = uvs.max(dim=0).values
            uv_range = (uv_max - uv_min).clamp(min=1e-8)
            uvs = (uvs - uv_min) / uv_range

            all_uvs[v_start:v_end] = uvs
            # Store aspect ratio for packing (proportional to 3D extent)
            chart_widths.append(uv_range[0].item())
            chart_heights.append(uv_range[1].item())

        # --- Shelf-based atlas packing ---
        uv_pad = max(padding_pixels / resolution, 0.001)

        # Sort charts by height descending for better shelf packing
        chart_order = sorted(range(num_charts), key=lambda i: -chart_heights[i])

        # Normalize chart sizes so total area roughly fits in a unit square
        total_area = sum(w * h for w, h in zip(chart_widths, chart_heights))
        if total_area < 1e-12:
            total_area = 1.0
        scale_factor = 0.9 / math.sqrt(total_area)  # ~90% fill target

        scaled_w = [chart_widths[i] * scale_factor for i in range(num_charts)]
        scaled_h = [chart_heights[i] * scale_factor for i in range(num_charts)]

        # Pack with shelves
        placements = [None] * num_charts  # (offset_x, offset_y, scale_x, scale_y)
        shelf_y = 0.0
        shelf_x = 0.0
        shelf_height = 0.0
        atlas_width = 0.0
        atlas_height = 0.0

        for idx in chart_order:
            w = scaled_w[idx] + 2 * uv_pad
            h = scaled_h[idx] + 2 * uv_pad

            # Start new shelf if this chart doesn't fit
            if shelf_x + w > 1.0 and shelf_x > 0:
                shelf_y += shelf_height
                shelf_x = 0.0
                shelf_height = 0.0

            placements[idx] = (shelf_x + uv_pad, shelf_y + uv_pad, scaled_w[idx], scaled_h[idx])
            shelf_x += w
            shelf_height = max(shelf_height, h)
            atlas_width = max(atlas_width, shelf_x)
            atlas_height = max(atlas_height, shelf_y + shelf_height)

        # Normalize so everything fits in [0, 1]
        if atlas_height > 1e-8:
            y_scale = 1.0 / atlas_height
        else:
            y_scale = 1.0
        if atlas_width > 1e-8:
            x_scale = 1.0 / atlas_width
        else:
            x_scale = 1.0
        norm_scale = min(x_scale, y_scale)

        # Apply placements to UVs
        for i in range(num_charts):
            v_start = chart_vertex_offset[i].item()
            v_end = chart_vertex_offset[i + 1].item()
            if v_end <= v_start:
                continue

            ox, oy, sw, sh = placements[i]
            scale = torch.tensor([sw * norm_scale, sh * norm_scale], device=device)
            offset = torch.tensor([ox * norm_scale, oy * norm_scale], device=device)
            all_uvs[v_start:v_end] = all_uvs[v_start:v_end] * scale + offset

        return chart_vertices, chart_faces, all_uvs, chart_vmap

    def uv_unwrap(
        self,
        compute_charts_kwargs: dict={},
        xatlas_compute_charts_kwargs: dict={},
        xatlas_pack_charts_kwargs: dict={},
        return_vmaps: bool=False,
        verbose: bool=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameterize the mesh using the accelerated mesh clustering and Xatlas

        Args:
            compute_charts_kwargs: a dictionary of options for the compute_charts function.
            xatlas_compute_charts_kwargs: a dictionary of options for the xatlas compute_charts function.
            xatlas_pack_charts_kwargs: a dictionary of options for the xatlas pack_charts function.
            return_vmaps: whether to return the vertex maps.
            verbose: whether to print the progress.

        Returns:
            A tuple of:
                - the vertex positions
                - the face indices
                - the uv coordinates
                - (optional) the map from the new vertex indices to the old vertex indices
        """
        self.remove_degenerate_faces()

        # 1. Fast GPU mesh clustering
        self.compute_charts(**compute_charts_kwargs)
        new_vertices, new_faces = self.read()
        num_charts, charts_id, chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset = self.read_atlas_charts()

        if verbose:
            print(f"Get {num_charts} clusters after fast clustering")

        # 2. UV parameterization
        use_gpu = bool(getattr(torch.version, 'hip', False))

        if use_gpu:
            # GPU-based PCA projection + shelf packing (fast, ROCm-friendly)
            if verbose:
                print("Using GPU UV parameterization (ROCm)")
            pack_res = xatlas_pack_charts_kwargs.get('resolution', 1024)
            pack_pad = xatlas_pack_charts_kwargs.get('padding', 1)
            vertices, faces, uvs, vmaps = self._gpu_uv_parameterize(
                new_vertices, num_charts, chart_vmap, chart_faces,
                chart_vertex_offset, chart_face_offset,
                padding_pixels=pack_pad, resolution=pack_res, verbose=verbose,
            )
            vertices = vertices.cpu()
            faces = faces.cpu()
            uvs = uvs.cpu()
            vmaps = vmaps.cpu()
        else:
            # Original xatlas path (CPU, for CUDA systems)
            xatlas_compute_charts_kwargs['verbose'] = verbose
            xatlas_pack_charts_kwargs['verbose'] = verbose
            chart_vertices = new_vertices[chart_vmap].cpu()
            chart_faces_cpu = chart_faces.cpu()
            chart_vertex_offset_cpu = chart_vertex_offset.cpu()
            chart_face_offset_cpu = chart_face_offset.cpu()
            chart_vmap_cpu = chart_vmap.cpu()

            xatlas = Atlas()
            chart_vmaps = []
            for i in tqdm(range(num_charts), desc="Adding clusters to xatlas", disable=not verbose):
                chart_faces_i = chart_faces_cpu[chart_face_offset_cpu[i]:chart_face_offset_cpu[i+1]] - chart_vertex_offset_cpu[i]
                chart_vertices_i = chart_vertices[chart_vertex_offset_cpu[i]:chart_vertex_offset_cpu[i+1]]
                chart_vmap_i = chart_vmap_cpu[chart_vertex_offset_cpu[i]:chart_vertex_offset_cpu[i+1]]
                chart_vmaps.append(chart_vmap_i)
                xatlas.add_mesh(chart_vertices_i, chart_faces_i)
            xatlas.compute_charts(**xatlas_compute_charts_kwargs)
            xatlas.pack_charts(**xatlas_pack_charts_kwargs)
            vmaps_list = []
            faces_list = []
            uvs_list = []
            cnt = 0
            for i in tqdm(range(num_charts), desc="Gathering results from xatlas", disable=not verbose):
                vmap, x_faces, x_uvs = xatlas.get_mesh(i)
                vmaps_list.append(chart_vmaps[i][vmap])
                faces_list.append(x_faces + cnt)
                uvs_list.append(x_uvs)
                cnt += vmap.shape[0]
            vmaps = torch.cat(vmaps_list, dim=0)
            vertices = new_vertices.cpu()[vmaps]
            faces = torch.cat(faces_list, dim=0)
            uvs = torch.cat(uvs_list, dim=0)

        out = [vertices, faces, uvs]
        if return_vmaps:
            out.append(vmaps)

        return tuple(out)
            