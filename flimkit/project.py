from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

#  Leica .sptw sub-folder names to search for tile PTUs 
_SPTW_CANDIDATES = [
    "{stem}.sptw",   # most common: scan-name.sptw
    "PTU.sptw",      # older Leica export style
    "FLIM.sptw",
]

PROJECT_FILENAME = "project.json"
DEFAULT_OUTPUT_SUBDIR = "output"


@dataclass
class ScanRecord:
    stem:          str
    scan_type:     str            # "fov" | "xlif"
    source_path:   str            # absolute path to .ptu or .xlif
    ptu_dir:       Optional[str] = None  # absolute path to .sptw folder (XLIF only)
    out_st:        Optional[str] = None  # base output dir last used (XLIF only)
    output_prefix: Optional[str] = None  # sv_out_fov prefix last used (FOV only)
    xlsx_path:     Optional[str] = None  # paired .xlsx file path (FOV only, e.g. file.ptu -> file.xlsx)

    #  derived helpers 

    @property
    def roi_clean(self) -> str:
        """Stem with spaces replaced — used by the stitch pipeline for subfolder naming."""
        return self.stem.replace(" ", "_")

    @property
    def session_path(self) -> Optional[Path]:
        """Return the fit session .npz path if it exists, else None."""
        if self.scan_type == "fov":
            p = Path(self.source_path)
            candidate = p.parent / f"{p.stem}.roi_session.npz"
        else:  # xlif
            if not self.out_st:
                return None
            candidate = Path(self.out_st) / self.roi_clean / "roi_session.npz"
        return candidate if candidate.exists() else None

    @property
    def phasor_session_path(self) -> Optional[Path]:
        """Return the phasor session .npz path if it exists, else None."""
        if self.scan_type != "fov":
            return None
        p = Path(self.source_path)
        candidate = p.parent / f"{p.stem}_phasor.npz"
        return candidate if candidate.exists() else None

    @property
    def has_session(self) -> bool:
        return self.session_path is not None

    @property
    def has_phasor_session(self) -> bool:
        return self.phasor_session_path is not None


class ProjectFile:
    """
    Represents a FLIMKit project (a folder of scans).

    Usage::

        pf = ProjectFile.load_or_create(Path("/path/to/experiment"))
        pf.save()   # writes project.json

    After a fit completes, call ``update_after_fit`` so the output location
    is remembered for next time::

        pf.update_after_fit("R 2", out_st="/path/to/experiment/output")
        pf.save()
    """

    def __init__(self, project_dir: Path):
        self.project_dir: Path = Path(project_dir).resolve()
        self.output_base: Path = self.project_dir / DEFAULT_OUTPUT_SUBDIR
        self.scans: Dict[str, ScanRecord] = {}

    # persistence

    @classmethod
    def load_or_create(cls, project_dir: Path) -> "ProjectFile":
        """
        Load project.json if it exists, then rescan the folder for new files.
        New files are added; existing records are not overwritten.
        """
        pf = cls(project_dir)
        json_path = pf.project_dir / PROJECT_FILENAME
        if json_path.exists():
            try:
                with open(json_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                ob = data.get("output_base")
                if ob:
                    pf.output_base = Path(ob)
                for stem, rec_dict in data.get("scans", {}).items():
                    pf.scans[stem] = ScanRecord(**rec_dict)
            except Exception as exc:
                # Corrupted project.json — start fresh, log the error
                print(f"[Project] Warning: could not read {json_path.name}: {exc}")
        pf._scan_folder()
        return pf

    def save(self):
        """Write (or overwrite) project.json in the project directory."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "output_base": str(self.output_base),
            "scans": {stem: asdict(rec) for stem, rec in self.scans.items()},
        }
        with open(self.project_dir / PROJECT_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    #  scan discovery

    def _scan_folder(self):
        """
        Walk project_dir (one level only) for .ptu and .xlif files.
        Adds new scans; leaves existing records untouched.
        For each PTU found, also checks for a paired .xlsx with the same name.
        """
        for ptu in sorted(self.project_dir.glob("*.ptu")):
            if ptu.name.startswith("._"):
                continue
            if ptu.stem not in self.scans:
                # Check for paired .xlsx file (same name, different extension)
                xlsx_file = self.project_dir / f"{ptu.stem}.xlsx"
                xlsx_path = str(xlsx_file) if xlsx_file.exists() and not xlsx_file.name.startswith("._") else None
                
                self.scans[ptu.stem] = ScanRecord(
                    stem          = ptu.stem,
                    scan_type     = "fov",
                    source_path   = str(ptu),
                    ptu_dir       = None,
                    out_st        = None,
                    output_prefix = None,
                    xlsx_path     = xlsx_path,
                )

        for xlif in sorted(self.project_dir.glob("*.xlif")):
            if xlif.name.startswith("._"):
                continue
            if xlif.stem not in self.scans:
                ptu_dir = self._find_sptw(xlif)
                self.scans[xlif.stem] = ScanRecord(
                    stem          = xlif.stem,
                    scan_type     = "xlif",
                    source_path   = str(xlif),
                    ptu_dir       = str(ptu_dir) if ptu_dir else None,
                    out_st        = str(self.output_base),
                    output_prefix = None,
                    xlsx_path     = None,
                )

    def _find_sptw(self, xlif: Path) -> Optional[Path]:
        """
        Locate the .sptw sub-folder that holds the tile PTUs for *xlif*.

        Tries, in order:
          1. <stem>.sptw  (most common Leica convention)
          2. PTU.sptw
          3. FLIM.sptw
        """
        base = xlif.parent
        for template in _SPTW_CANDIDATES:
            candidate = base / template.format(stem=xlif.stem)
            if candidate.is_dir():
                return candidate
        return None

    #  post-fit update 

    def update_after_fit(
        self,
        stem: str,
        *,
        out_st: Optional[str] = None,
        output_prefix: Optional[str] = None,
        ptu_dir: Optional[str] = None,
    ):
        """
        Record the output locations used in a completed fit so the browser
        can find the session file on the next launch.

        Call this after a fit completes, then call ``save()``.
        """
        rec = self.scans.get(stem)
        if rec is None:
            return
        if out_st is not None:
            rec.out_st = out_st
        if output_prefix is not None:
            rec.output_prefix = output_prefix
        if ptu_dir is not None:
            rec.ptu_dir = ptu_dir

    def update_after_phasor(self, stem: str):
        """No-op placeholder — phasor sessions are discovered by file convention."""
        pass

    #  convenience

    def default_out_st(self, stem: str) -> str:
        """
        Return the base output directory for an XLIF scan.
        Uses the stored value if available, otherwise the project output_base.
        """
        rec = self.scans.get(stem)
        if rec and rec.out_st:
            return rec.out_st
        return str(self.output_base)

    def default_output_prefix(self, stem: str) -> str:
        """Return the output prefix for a FOV scan (defaults to PTU base name)."""
        rec = self.scans.get(stem)
        if rec and rec.output_prefix:
            return rec.output_prefix
        return stem

    def sorted_scans(self):
        """Yield (stem, ScanRecord) sorted alphabetically."""
        yield from sorted(self.scans.items())
