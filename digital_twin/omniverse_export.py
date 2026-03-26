import numpy as np

def _grid_to_obj(x, y, z, name="Object"):
    """Converts 2D numpy arrays of x, y, z coordinates (from meshgrid) to OBJ format string."""
    obj_str = f"o {name}\n"
    rows, cols = x.shape
    
    # Write vertices
    for i in range(rows):
        for j in range(cols):
            obj_str += f"v {x[i,j]:.6f} {y[i,j]:.6f} {z[i,j]:.6f}\n"
            
    # Write faces (quads split into two triangles)
    # The vertex index in OBJ is 1-based.
    # We will build them keeping a global offset in mind if combining, but here we return a standalone string
    # where vertices start at 1. We will manage offsets later.
    faces_str = ""
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Vertex indices
            v1 = i * cols + j + 1
            v2 = i * cols + (j + 1) + 1
            v3 = (i + 1) * cols + j + 1
            v4 = (i + 1) * cols + (j + 1) + 1
            
            # Triangle 1
            faces_str += f"f {v1} {v2} {v3}\n"
            # Triangle 2
            faces_str += f"f {v2} {v4} {v3}\n"
            
    return obj_str + faces_str, rows * cols

def export_to_obj(twin_data):
    """
    Exports the generated Digital Twin data to an OBJ file for NVIDIA Omniverse.
    Combines the distinct meshes (Uterus, Ovaries, Tubes) into one coherent .obj
    """
    final_obj = "# Exported from Endometriosis Digital Twin for NVIDIA Omniverse\n"
    vertex_offset = 0
    
    def add_mesh(x, y, z, name):
        nonlocal final_obj, vertex_offset
        obj_chunk, num_verts = _grid_to_obj(x, y, z, name)
        
        # Adjust face indices in this chunk based on the current vertex_offset
        if vertex_offset > 0:
            lines = obj_chunk.split('\n')
            adjusted_lines = []
            for line in lines:
                if line.startswith('f '):
                    parts = line.split()
                    adjusted_parts = ['f'] + [str(int(idx) + vertex_offset) for idx in parts[1:]]
                    adjusted_lines.append(' '.join(adjusted_parts))
                else:
                    adjusted_lines.append(line)
            obj_chunk = '\n'.join(adjusted_lines)
            
        final_obj += obj_chunk + "\n"
        vertex_offset += num_verts

    # Uterus
    u_x, u_y, u_z = twin_data['uterus']
    add_mesh(u_x, u_y, u_z, "Uterus")
    
    # Left Ovary
    lo_x, lo_y, lo_z = twin_data['left_ovary']
    add_mesh(lo_x, lo_y, lo_z, "Left_Ovary")
    
    # Right Ovary
    ro_x, ro_y, ro_z = twin_data['right_ovary']
    add_mesh(ro_x, ro_y, ro_z, "Right_Ovary")
    
    # Left Tube
    lt_x, lt_y, lt_z = twin_data['left_tube']
    add_mesh(lt_x, lt_y, lt_z, "Left_Fallopian_Tube")
    
    # Right Tube
    rt_x, rt_y, rt_z = twin_data['right_tube']
    add_mesh(rt_x, rt_y, rt_z, "Right_Fallopian_Tube")
    
    return final_obj

def export_lesions_to_usd_ascii(twin_data):
    """
    Omniverse prefers point clouds in USD. 
    This generates a standalone .usda file just for the lesions, combining current and future.
    """
    lx, ly, lz, lc = twin_data['lesions']
    fx, fy, fz, fc = twin_data.get('future_lesions', ([], [], [], []))
    
    all_x = list(lx) + list(fx)
    all_y = list(ly) + list(fy)
    all_z = list(lz) + list(fz)
    
    if len(all_x) == 0:
        return ""
        
    points_str = ", ".join([f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in zip(all_x, all_y, all_z)])
    
    usda = f'''#usda 1.0
(
    defaultPrim = "Lesions"
    upAxis = "Z"
)

def Points "Lesions"
{{
    point3f[] points = [{points_str}]
    float[] widths = [0.2]
}}
'''
    return usda
