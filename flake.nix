{
  description = "Stat estimation for Torn City";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      # Adds `statstimate` to every Python package set (python3Packages, python312Packages, ...).
      overlays.default = final: prev: {
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
          (pyFinal: pyPrev: { statstimate = pyFinal.callPackage ./nix/package.nix { }; })
        ];
      };

      packages = forAllSystems (pkgs: rec {
        statstimate = pkgs.python3Packages.callPackage ./nix/package.nix { };
        default = statstimate;
      });

      checks = forAllSystems (
        pkgs:
        let
          python = pkgs.python3.withPackages (ps: [ (ps.callPackage ./nix/package.nix { }) ]);
        in
        {
          package = self.packages.${pkgs.stdenv.hostPlatform.system}.statstimate;

          # The unit tests use stub models; this loads the real bundled ones with
          # nixpkgs' xgboost to make sure that version can read them.
          bundled-models = pkgs.runCommand "statstimate-bundled-models" { } ''
            ${python}/bin/python -c "import statstimate; statstimate.Estimator()"
            touch $out
          '';
        }
      );

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: [
              ps.numpy
              ps.xgboost
              ps.pytest
              ps.pytest-sugar
              ps.black
              ps.isort
            ]))
            pkgs.uv
          ];
          shellHook = ''
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
          '';
        };
      });
    };
}
