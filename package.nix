{
  lib,
  buildPythonPackage,
  hatchling,
  numpy,
  xgboost,
  pytestCheckHook,
}:

let
  # Read the version from src/statstimate/__init__.py so it never drifts from PyPI.
  initFile = builtins.readFile ../src/statstimate/__init__.py;
  versionLine = lib.findFirst (lib.hasPrefix "__version__") null (lib.splitString "\n" initFile);
  version = builtins.elemAt (lib.splitString "\"" versionLine) 1;
in
buildPythonPackage {
  pname = "statstimate";
  inherit version;
  pyproject = true;

  src = lib.cleanSource ../.;

  build-system = [ hatchling ];

  # The PyPI extras ([cpu] -> xgboost-cpu, [gpu] -> xgboost) don't map onto nixpkgs:
  # there is one xgboost package, CPU by default and CUDA-enabled when
  # nixpkgs is configured with `cudaSupport = true`. So depend on it directly.
  dependencies = [
    numpy
    xgboost
  ];

  nativeCheckInputs = [ pytestCheckHook ];

  pythonImportsCheck = [ "statstimate" ];

  meta = {
    description = "Stat estimation for Torn City";
    homepage = "https://github.com/Tornium/statstimate";
    changelog = "https://github.com/Tornium/statstimate/blob/master/CHANGELOG.md";
    license = lib.licenses.gpl3Plus;
    platforms = lib.platforms.unix;
  };
}
