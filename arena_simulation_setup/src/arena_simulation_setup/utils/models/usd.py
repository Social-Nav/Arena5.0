import os
import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Collection
from pathlib import Path

import aiofiles
import yaml

from . import Model, ModelProvider, ModelType


class ModelProvider_USD(ModelProvider.provides(ModelType.USD)):
    @classmethod
    def _resolve_redirect_path(cls, model_dir: Path, params_path: Path, redirect: str) -> Path:
        redirect_path = Path(redirect)
        if redirect_path.is_absolute():
            return redirect_path

        # Prefer resolving relative to model_params.yaml real location so this
        # also works when model_params.yaml is a symlink in install/share.
        params_based = (params_path.resolve().parent / redirect_path).resolve()
        if params_based.exists():
            return params_based

        # Fallback: resolve relative to model directory.
        return (model_dir / redirect_path).resolve()

    @classmethod
    async def load(cls, model_dir, model, loader_args) -> Model:
        model_paths = (
            model_dir / f"{model}.usdz",
            model_dir / f"{model}.usd",
            model_dir / f"{model}.usda",
            model_dir / f"{model}.usdc",
        )

        found = next(filter(os.path.exists, model_paths), None)
        if found is None:
            params_path = model_dir / "model_params.yaml"
            if params_path.exists():
                with open(params_path) as f:
                    params = yaml.safe_load(f) or {}

                if isinstance(params, dict):
                    redirect = params.get("usd_redirect")
                    if isinstance(redirect, str) and redirect.strip():
                        redirected = cls._resolve_redirect_path(model_dir, params_path, redirect.strip())
                        if redirected.exists():
                            found = redirected
                        else:
                            raise FileNotFoundError(
                                f"usd_redirect points to missing file: {redirected} "
                                f"(from {params_path})"
                            )

        if found is None:
            raise FileNotFoundError(f"USD model for {model} not found in {model_dir}")
        return Model(
            type=ModelType.USD,
            name=model,
            description="",  # TODO add bytes compat
            path=found
        )

    @classmethod
    def convertable(cls) -> Collection[ModelType]:
        return (ModelType.SDF,)

    @classmethod
    async def convert(cls, model_dir, model, loader_args) -> Model | None:
        if model.type is ModelType.SDF:
            try:
                # print(model_dir)
                model_path = model.path
                model_dir = model_path.parent
                async with aiofiles.open(model_path, 'r') as f:
                    tree = ET.ElementTree(ET.fromstring(await f.read()))
                root = tree.getroot()
                assert root is not None
                # First pass: resolve package:// URIs
                model_uri_pattern = re.compile(r'^model://([^/]+)(.*)$')
                package_uri_pattern = re.compile(r'^package://([^/]+)(.*)$')
                for uri_elem in root.iter():
                    if uri_elem.text:
                        text = uri_elem.text.strip()
                        match = model_uri_pattern.match(text)
                        if match:
                            package_name = match.group(1)
                            remaining_path = match.group(2)
                            # Get the absolute path for the package share directory.
                            # Replace the package URI with the resolved directory plus remaining path.
                            new_uri = model_dir / remaining_path
                            print(new_uri)
                            # uri_elem.text = new_uri
                            if str(new_uri).lower().endswith('.dae') and new_uri.is_file():
                                uri_elem.text = str(await process_dae(new_uri, model_dir))
                            elif str(new_uri).lower().endswith('.obj') and new_uri.is_file():
                                uri_elem.text = str(await process_obj(new_uri, model_dir))
                        else:
                            match = package_uri_pattern.match(text)
                            if match:
                                package_name = match.group(1)
                                remaining_path = match.group(2)
                                # Get the absolute path for the package share directory.
                                # Replace the package URI with the resolved directory plus remaining path.
                                new_uri = model_dir / remaining_path
                                print(new_uri)
                                if str(new_uri).lower().endswith('.dae') and new_uri.is_file():
                                    uri_elem.text = str(await process_dae(new_uri, model_dir))
                                elif str(new_uri).lower().endswith('.obj') and new_uri.is_file():
                                    uri_elem.text = str(await process_obj(new_uri, model_dir))
                model_path = model_dir / model.name / "usd" / f"{model.name}.usd"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                if model_path.is_symlink() and not model_path.exists():  # broken symlink
                    model_path.unlink()

                import arena_bringup
                ARENA_DIR = arena_bringup.get_arena_dir()

                env = os.environ.copy()
                env['ARENA_WS_DIR'] = ARENA_DIR
                async with aiofiles.tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
                    ser = ET.tostring(root, encoding="unicode", method="xml")
                    await f.write(ser)
                    await f.flush()
                    print("Temporary URDF file for converter:", f.name)
                    subprocess.check_output(

                        [
                            f'{ARENA_DIR}/_meta/tools/sdf2usd',
                            f.name,
                            model_path
                        ],
                        env=env,
                        # shell=True,
                    )

                from pxr import Usd
                stage = Usd.Stage.Open(model_path)  # type: ignore
                for prim in stage.Traverse():
                    if prim.GetTypeName() == "Xform":
                        first_xform_prim = prim
                        break
                else:
                    raise RuntimeError('no xform prim found')
                prim_path = first_xform_prim.GetPath()
                # print(prim_path)
                stage.SetDefaultPrim(first_xform_prim)
                root_layer = stage.GetRootLayer()
                root_layer.Save()
                return await cls.load(model_dir, model.name, loader_args)

            except Exception:
                raise

        return None


async def process_dae(dae_file, package_dir) -> Path:
    """
    Load a .dae file, update its <init_from> elements by replacing any leading
    '../' with the package_dir, then write to a temporary file and return its path.
    """
    import collada
    file = collada.Collada(dae_file)
    tree = file.xmlnode
    root = tree.getroot()
    assert root is not None
    for init_elem in root.iterfind('.//init_from'):
        print(init_elem)
        if init_elem.text:
            text = init_elem.text.strip()
            if text.startswith("../"):
                # Remove all leading "../" segments
                rel_path = text
                while rel_path.startswith("../"):
                    rel_path = rel_path[3:]
                # Create a new absolute path using the package directory
                new_text = os.path.join(package_dir, rel_path)
                init_elem.text = new_text

    async with aiofiles.tempfile.NamedTemporaryFile(mode="wb", suffix=".dae", delete=False) as tmp_file:
        # Write the XML tree to the temporary file.
        ser = ET.tostring(root, encoding="utf-8", method="xml", xml_declaration=True)
        await tmp_file.write(ser)
    return Path(tmp_file.name)

    # Write the updated .dae file to a temporary file


async def process_obj(obj_file, package_dir) -> Path:
    """
    Read an .obj file as text and update any .png file references.
    For any found relative .png path (e.g. starting with "../"), remove the
    relative segments and prepend the package_dir. The modified file is saved
    to a temporary file whose path is returned.
    """
    try:
        async with aiofiles.open(obj_file, 'r', encoding='utf-8') as f:
            content = await f.read()
    except Exception as e:
        print(f"Error reading {obj_file}: {e}")
        return obj_file  # fallback: return original file if error occurs

    # Regex to match .png filenames (non-space characters ending in .png)
    png_pattern = re.compile(r'(?P<path>\S+\.png)')
    mtl_patter = re.compile(r'(?P<path>\S+\.mtl)')

    def replace_png(match):
        path = match.group("path")
        # If already absolute, do nothing.
        if os.path.isabs(path):
            return path
        # Remove any leading '../' segments
        while path.startswith("../"):
            path = path[3:]
        # Return the absolute path by joining with the package directory
        return os.path.join(package_dir, path)

    new_content = png_pattern.sub(replace_png, content)

    # Write the updated .obj file to a temporary file
    async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix='.obj', mode='w', encoding='utf-8') as temp_file:
        await temp_file.write(new_content)
    return Path(temp_file.name)
