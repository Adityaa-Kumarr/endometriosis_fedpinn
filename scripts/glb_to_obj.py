"""Convert uterus .glb to .obj using trimesh. Run with: py glb_to_obj.py"""
import os
import sys

def main():
    glb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uterus .glb")
    if len(sys.argv) > 1:
        glb_path = sys.argv[1]
    if not os.path.isfile(glb_path):
        print(f"File not found: {glb_path}")
        sys.exit(1)
    try:
        import trimesh
    except ImportError:
        print("Install trimesh first: pip install trimesh")
        sys.exit(1)
    out_path = glb_path.replace(".glb", ".obj").replace(".glTF", ".obj")
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    print(f"Loading {glb_path} ...")
    mesh = trimesh.load(glb_path, force="mesh")
    if hasattr(mesh, "geometry"):
        mesh = list(mesh.geometry.values())[0]
    mesh.export(out_path)
    print(f"Exported: {out_path}")

if __name__ == "__main__":
    main()
