from __future__ import annotations
import os
from typing import List, Tuple, Optional, Dict
import pygame
import torch
import torch.nn as nn
import numpy as np

import config
from simulation.stats import TEAM_RED, TEAM_BLUE
from .camera import Camera
from engine.agent_registry import COL_ALIVE, COL_TEAM, COL_UNIT
from agent.mutation import pick_mutants, mutate_model_inplace

FONT_NAME = "consolas"

# ---------- colors ----------
COLORS = {
    "bg":        (18, 20, 24),
    "hud_bg":    (12, 12, 14),
    "side_bg":   (16, 18, 22),
    "grid":      (42, 45, 52),
    "border":    (64, 68, 76),
    "wall":      (78, 82, 92),
    "empty":     (22, 24, 28),

    # team base colors
    "red":       (220, 80, 80),
    "blue":      (80, 120, 240),

    # team-by-unit variants (archers slightly lighter)
    "red_soldier":  (220, 80, 80),
    "red_archer":   (245, 135, 135),
    "blue_soldier": (80, 120, 240),
    "blue_archer":  (130, 165, 255),

    # archer glyph overlay (high-contrast ring)
    "archer_glyph": (245, 230, 90),

    "marker":    (242, 228, 92),
    "text":      (230, 230, 230),
    "text_dim":  (180, 186, 194),
    "green":     (110, 200, 140),
    "warn":      (240, 160, 90),
    "bar_bg":    (38, 42, 48),
    "bar_fg":    (90, 200, 130),
}

# RGBA overlays (used on SRCALPHA surfaces)
OVERLAY_HEAL = (90, 200, 140, 60)     # faint green fill per heal tile
OVERLAY_CP   = (210, 210, 230, 48)    # neutral fill for capture patches
OUTLINE_RED  = (220, 80, 80, 160)
OUTLINE_BLUE = (80, 120, 240, 160)
OUTLINE_NEU  = (160, 160, 170, 120)

def _mk_font(sz: int) -> pygame.font.Font:
    try:
        return pygame.font.SysFont(FONT_NAME, sz)
    except Exception:
        return pygame.font.Font(None, sz)

def _center_window() -> None:
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

# ---------- model introspection ----------
def _linear_shape_list(model: nn.Module) -> List[Tuple[int, int]]:
    shapes = []
    try:
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                shapes.append((module.in_features, module.out_features))
    except Exception:
        pass
    if not shapes:
        try:
            for k, v in model.state_dict().items():
                if "weight" in k and v.ndim == 2:
                    shapes.append((v.shape[1], v.shape[0]))
        except Exception:
            pass
    return shapes

def _format_mlp_shape(shapes: List[Tuple[int, int]]) -> str:
    if not shapes:
        return "unknown"
    dims = [shapes[0][0]] + [o for (_, o) in shapes]
    return "→".join(str(d) for d in dims)

def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# ======================================================================
# Viewer
# ======================================================================
class Viewer:
    """
    Controls:
      Pan = WASD / Arrow keys    |   Zoom = Mouse Wheel
      Select = Left Click        |   Mark/Unmark = M (≤10)
      Copy selected brain = C    |   Mutate ~10% alive = E
      Quit = Esc
    """

    def __init__(self, grid: torch.Tensor, cell_size: Optional[int] = None, show_grid: bool = True):
        _center_window()
        pygame.init()
        pygame.display.set_caption("final_war_sim")

        self.grid = grid
        self.cell = int(cell_size or config.CELL_SIZE)
        H, W = grid.size(1), grid.size(2)

        # camera
        self.cam = Camera(self.cell, W, H)

        # layout constants
        self.margin = 8
        self.hud_h = 126
        self.side_min_w = 260
        self.side_max_w = 420

        # window (with caps)
        max_w, max_h = 1200, 800
        world_px_w = W * self.cell
        world_px_h = H * self.cell
        init_w = min(max_w, max(800, world_px_w + self.side_min_w + 3 * self.margin))
        init_h = min(max_h, max(520, world_px_h + 2 * self.margin + self.hud_h))
        self.Wpix, self.Hpix = int(init_w), int(init_h)

        self.screen = pygame.display.set_mode((self.Wpix, self.Hpix), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = _mk_font(16)
        self.font_small = _mk_font(13)
        self.font_big = _mk_font(18)

        self.selected_id: Optional[int] = None
        self.marked: List[int] = []
        self.prev_occ: Optional[np.ndarray] = None
        self.prev_unit: Optional[np.ndarray] = None
        self.show_grid = show_grid
        self.registry = None  # set in run()

        # prebuilt zone overlay geometry (CPU cache)
        self._zone_cache: Optional[Dict[str, object]] = None

    # ---------- dynamic layout ----------
    def _side_width(self) -> int:
        target = int(self.Wpix * 0.27)
        return max(self.side_min_w, min(self.side_max_w, target))

    def _world_rect(self) -> pygame.Rect:
        side_w = self._side_width()
        vw = max(64, self.Wpix - side_w - 3 * self.margin)
        vh = max(64, self.Hpix - self.hud_h - 2 * self.margin)
        return pygame.Rect(self.margin, self.margin, vw, vh)

    def _side_rect(self) -> pygame.Rect:
        side_w = self._side_width()
        x = self.Wpix - side_w - self.margin
        y = self.margin
        h = max(64, self.Hpix - self.hud_h - 2 * self.margin)
        return pygame.Rect(x, y, side_w, h)

    def _hud_rect(self) -> pygame.Rect:
        return pygame.Rect(0, self.Hpix - self.hud_h, self.Wpix, self.hud_h)

    # ---------- registry helpers ----------
    def _get_brain(self, agent_id: int):
        if self.registry is None:
            return None
        brains = getattr(self.registry, "brains", None)
        if brains is None:
            return None
        if isinstance(brains, dict):
            return brains.get(agent_id, None)
        try:
            return brains[agent_id] if 0 <= agent_id < len(brains) else None
        except Exception:
            return None

    def _get_generation(self, agent_id: int) -> Optional[int]:
        if self.registry is None:
            return None
        get_gen = getattr(self.registry, "get_agent_generation", None)
        if callable(get_gen):
            try:
                return int(get_gen(agent_id))
            except Exception:
                return None
        gens = getattr(self.registry, "generations", None)
        if isinstance(gens, dict):
            return gens.get(agent_id, None)
        return None

    # ---------- mutation ----------
    def _do_mutate_10pct(self) -> None:
        if not self.registry:
            print("[viewer] mutate skipped: registry not set")
            return
        data = self.registry.agent_data
        alive_idx = (data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)
        if alive_idx.numel() == 0:
            print("[viewer] mutate skipped: no alive agents")
            return

        chosen = pick_mutants(alive_idx, fraction=0.10)
        if chosen.numel() == 0:
            print("[viewer] mutate skipped: picker returned empty set")
            return

        self.registry.apply_mutations(chosen, mutate_model_inplace)
        print(f"[viewer] mutated {int(chosen.numel())}/{int(alive_idx.numel())} alive (~10%)")

    # ---------- input ----------
    def _handle_input(self) -> bool:
        running = True
        keys = pygame.key.get_pressed()
        pan_speed = 10.0 / max(1.0, float(self.cam.cell_px))

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  self.cam.pan(-pan_speed, 0)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.cam.pan(+pan_speed, 0)
        if keys[pygame.K_w] or keys[pygame.K_UP]:    self.cam.pan(0, -pan_speed)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  self.cam.pan(0, +pan_speed)

        wrect = self._world_rect()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.VIDEORESIZE:
                self.Wpix, self.Hpix = max(800, ev.w), max(520, ev.h)
                self.screen = pygame.display.set_mode((self.Wpix, self.Hpix), pygame.RESIZABLE)
                self.prev_occ = None
                self.prev_unit = None

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_c and self.selected_id is not None:
                    brain = self._get_brain(self.selected_id)
                    if brain is not None:
                        try:
                            torch.save(brain.state_dict(), "copied_brain.pth")
                            print("[viewer] saved selected brain -> copied_brain.pth")
                        except Exception as e:
                            print(f"[viewer] save failed: {e}")
                elif ev.key == pygame.K_m and self.selected_id is not None:
                    if self.selected_id in self.marked:
                        self.marked.remove(self.selected_id)
                    elif len(self.marked) < 10:
                        self.marked.append(self.selected_id)
                elif ev.key == pygame.K_e:
                    self._do_mutate_10pct()

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1 and wrect.collidepoint(ev.pos):  # select
                    lx = ev.pos[0] - wrect.x
                    ly = ev.pos[1] - wrect.y
                    gx, gy = self.cam.screen_to_world(lx, ly)
                    gy = max(0, min(int(gy), self.grid.size(1) - 1))
                    gx = max(0, min(int(gx), self.grid.size(2) - 1))
                    with torch.no_grad():
                        agent_id = int(self.grid[2, gy, gx].item())
                        self.selected_id = agent_id if agent_id >= 0 else None
                elif ev.button == 4:  # wheel up
                    self.cam.zoom_at(1.12); self.prev_occ = None; self.prev_unit = None
                elif ev.button == 5:  # wheel down
                    self.cam.zoom_at(1/1.12); self.prev_occ = None; self.prev_unit = None

        return running

    # ---------- drawing helpers ----------
    def _hud_text(self, text: str, x: int, y: int, color=COLORS["text"]) -> None:
        self.screen.blit(self.font.render(text, True, color), (x, y))

    def _hud_text_small(self, text: str, x: int, y: int, color=COLORS["text_dim"]) -> None:
        self.screen.blit(self.font_small.render(text, True, color), (x, y))

    def _draw_archer_glyph(self, surf: pygame.Surface, cx: int, cy: int, c: int) -> None:
        """Draw a high-contrast archer ring inside the cell.
        Scales with zoom, gracefully disappears for tiny cells.
        """
        if c < 5:
            return
        color = COLORS["archer_glyph"]
        # Center of cell
        center = (cx + c // 2, cy + c // 2)
        # Outer radius leaves 1px margin; thickness scales with cell size
        radius = max(2, (c // 2) - 1)
        width = max(1, c // 6)
        try:
            pygame.draw.circle(surf, color, center, radius, width)
        except Exception:
            # Fallback: small inner square if circle fails for any reason
            pad = max(1, c // 4)
            pygame.draw.rect(surf, color, pygame.Rect(cx + pad, cy + pad, max(1, c - 2 * pad), max(1, c - 2 * pad)), width=max(1, width // 2))

    def _draw_cell(self, surf: pygame.Surface, x: int, y: int, occ_val: int, unit_val: int) -> None:
        # occ: 0 empty, 1 wall, 2 red, 3 blue
        if occ_val == 1:
            color = COLORS["wall"]
        elif occ_val == 2:
            color = COLORS["red_archer"] if unit_val == 2 else COLORS["red_soldier"] if unit_val == 1 else COLORS["red"]
        elif occ_val == 3:
            color = COLORS["blue_archer"] if unit_val == 2 else COLORS["blue_soldier"] if unit_val == 1 else COLORS["blue"]
        else:
            color = COLORS["empty"]

        cx, cy = self.cam.world_to_screen(x, y)
        c = self.cam.cell_px
        wrect = self._world_rect()
        cx += wrect.x; cy += wrect.y
        if cx >= wrect.x - c and cy >= wrect.y - c and cx < wrect.right and cy < wrect.bottom:
            cell_rect = pygame.Rect(cx, cy, c, c)
            surf.fill(color, cell_rect)
            # Archer highlight: draw a bright ring overlay
            if (occ_val == 2 or occ_val == 3) and unit_val == 2:
                self._draw_archer_glyph(surf, cx, cy, c)

    def _draw_grid_lines(self, surf: pygame.Surface) -> None:
        if not self.show_grid or self.cam.cell_px < 6:
            return
        c = self.cam.cell_px
        wrect = self._world_rect()

        ax, ay = self.cam.world_to_screen(0, 0)
        ax += wrect.x; ay += wrect.y
        off_x = (c - ((ax - wrect.x) % c)) % c
        off_y = (c - ((ay - wrect.y) % c)) % c

        x = wrect.x + off_x
        while x < wrect.right:
            pygame.draw.line(surf, COLORS["grid"], (x, wrect.y), (x, wrect.bottom), width=1)
            x += c
        y = wrect.y + off_y
        while y < wrect.bottom:
            pygame.draw.line(surf, COLORS["grid"], (wrect.x, y), (wrect.right, y), width=1)
            y += c

        pygame.draw.rect(surf, COLORS["border"], wrect, width=2)

    def _draw_markers(self, surf: pygame.Surface, cpu_id: torch.Tensor) -> None:
        c = self.cam.cell_px
        wrect = self._world_rect()
        id_np = cpu_id.numpy()
        for aid in self.marked:
            ys, xs = (id_np == aid).nonzero()
            if len(xs) == 0:
                continue
            x, y = int(xs[0]), int(ys[0])
            cx, cy = self.cam.world_to_screen(x, y)
            cx += wrect.x; cy += wrect.y
            pygame.draw.rect(surf, COLORS["marker"], pygame.Rect(cx, cy, c, c), width=max(1, c // 8))

    # ---------- zone cache ----------
    def _build_zone_overlay_cache(self, engine) -> None:
        """Build static CPU-side geometry for overlays to avoid per-frame transfers."""
        self._zone_cache = {"heal_tiles": [], "cp_rects": []}
        zones = getattr(engine, "zones", None)
        if zones is None:
            return
        try:
            # heal tiles
            if getattr(zones, "heal_mask", None) is not None:
                hm = zones.heal_mask.detach().cpu().numpy()
                ys, xs = np.nonzero(hm)
                self._zone_cache["heal_tiles"] = list(zip(xs.tolist(), ys.tolist()))
            # capture rects (tight bounds of masks)
            for m in getattr(zones, "cp_masks", []) or []:
                mm = m.detach().cpu().numpy()
                ys, xs = np.nonzero(mm)
                if xs.size == 0:
                    continue
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                self._zone_cache["cp_rects"].append((x0, y0, x1, y1))
        except Exception as e:
            print(f"[viewer] WARN: zone cache build failed: {e}")
            self._zone_cache = {"heal_tiles": [], "cp_rects": []}

    # ---------- zone overlays ----------
    def _draw_zone_overlays(self, cpu_occ_np: np.ndarray) -> None:
        if not self._zone_cache:
            return

        wrect = self._world_rect()
        c = self.cam.cell_px
        overlay = pygame.Surface((wrect.width, wrect.height), pygame.SRCALPHA)

        # heal tiles
        for x, y in self._zone_cache["heal_tiles"]:
            cx, cy = self.cam.world_to_screen(int(x), int(y))
            if 0 <= cx < wrect.width and 0 <= cy < wrect.height:
                overlay.fill(OVERLAY_HEAL, pygame.Rect(cx, cy, c, c))

        # capture rects
        for (x0, y0, x1, y1) in self._zone_cache["cp_rects"]:
            cx0, cy0 = self.cam.world_to_screen(x0, y0)
            cx1, cy1 = self.cam.world_to_screen(x1, y1)
            rect = pygame.Rect(cx0, cy0, max(1, cx1 - cx0), max(1, cy1 - cy0))
            overlay.fill(OVERLAY_CP, rect)

            # determine leader from current occupancy
            patch = cpu_occ_np[y0:y1, x0:x1]
            red_cnt = int((patch == 2).sum())
            blue_cnt = int((patch == 3).sum())
            if red_cnt > blue_cnt:
                border_col = OUTLINE_RED;   label = ("R", COLORS["red"])
            elif blue_cnt > red_cnt:
                border_col = OUTLINE_BLUE;  label = ("B", COLORS["blue"])
            else:
                border_col = OUTLINE_NEU;   label = ("–", COLORS["text_dim"])

            pygame.draw.rect(overlay, border_col, rect, width=max(1, c // 2))
            lab = self.font_small.render(label[0], True, label[1])
            overlay.blit(lab, lab.get_rect(center=(rect.x + rect.w // 2, rect.y + rect.h // 2)))

        self.screen.blit(overlay, (wrect.x, wrect.y))

    def _draw_world(self, surf: pygame.Surface, cur_occ_np: np.ndarray, cur_unit_np: np.ndarray) -> None:
        H, W = cur_occ_np.shape
        wrect = self._world_rect()
        surf.fill(COLORS["bg"], wrect)

        full = (
            self.prev_occ is None
            or self.prev_unit is None
            or (self.cam.zoom != 1.0)
            or (self.cam.cell_px != self.cell)
        )
        if full:
            for y in range(H):
                for x in range(W):
                    self._draw_cell(surf, x, y, int(cur_occ_np[y, x]), int(cur_unit_np[y, x]))
        else:
            changed = np.where((self.prev_occ != cur_occ_np) | (self.prev_unit != cur_unit_np))
            if changed[0].size > 0:
                for y, x in zip(changed[0], changed[1]):
                    self._draw_cell(surf, int(x), int(y), int(cur_occ_np[y, x]), int(cur_unit_np[y, x]))
        self.prev_occ = cur_occ_np.copy()
        self.prev_unit = cur_unit_np.copy()

    # ---------- agent helpers ----------
    def _agent_current_xy(self, cpu_id: torch.Tensor, agent_id: int) -> Optional[Tuple[int, int]]:
        ys, xs = (cpu_id == agent_id).nonzero(as_tuple=True)
        if ys.numel() == 0:
            return None
        return (int(xs[0].item()), int(ys[0].item()))

    # ---------- side info panel ----------
    def _draw_side_panel(self, cpu_id: torch.Tensor) -> None:
        srect = self._side_rect()
        pygame.draw.rect(self.screen, COLORS["side_bg"], srect)
        pygame.draw.rect(self.screen, COLORS["border"], srect, width=2)

        pad = 10
        x = srect.x + pad
        y = srect.y + pad

        self.screen.blit(self.font_big.render("Agent Info", True, COLORS["text"]), (x, y))
        y += 30

        if self.selected_id is None:
            self._hud_text_small("No agent selected.", x, y); y += 20
            self._hud_text_small("Click an agent in the", x, y); y += 16
            self._hud_text_small("world to inspect it.", x, y); y += 24
            self._hud_text_small("Tips:", x, y, COLORS["green"]); y += 18
            self._hud_text_small("• M to mark up to 10", x, y); y += 16
            self._hud_text_small("• C to copy brain", x, y); y += 16
            self._hud_text_small("• E to mutate ~10%", x, y); y += 16
            self._hud_text_small("• Archers have a yellow ring", x, y); y += 16
            return

        aid = self.selected_id
        self._hud_text(f"ID: {aid}", x, y, COLORS["green"]); y += 24

        # live position + HP (from grid), team and alive flag (from registry)
        pos = self._agent_current_xy(cpu_id, aid)
        hp_txt = "n/a"
        hp_ratio = 0.0
        if pos is not None:
            px, py = pos
            self._hud_text_small(f"Pos: ({px}, {py})", x, y); y += 18
            with torch.no_grad():
                hp = float(self.grid[1, py, px].item())
                hp_txt = f"{hp:.1f}"
                hp_ratio = max(0.0, min(1.0, hp / float(config.MAX_HP)))
        else:
            self._hud_text_small("Pos: off-grid", x, y); y += 18

        # HP bar
        bar_w = srect.width - 2 * pad
        bar_h = 10
        pygame.draw.rect(self.screen, COLORS["bar_bg"], pygame.Rect(x, y, bar_w, bar_h))
        pygame.draw.rect(self.screen, COLORS["bar_fg"], pygame.Rect(x, y, int(bar_w * hp_ratio), bar_h))
        y += bar_h + 4
        self._hud_text_small(f"HP: {hp_txt} / {config.MAX_HP}", x, y); y += 22

        # Team + alive flag
        team_txt = "?"
        alive_txt = "?"
        if self.registry is not None and 0 <= aid < self.registry.agent_data.size(0):
            data = self.registry.agent_data
            alive = bool(data[aid, COL_ALIVE].item() > 0.5)
            alive_txt = "alive" if alive else "dead"
            tval = float(data[aid, COL_TEAM].item())
            team_txt = "RED" if tval == 2.0 else ("BLUE" if tval == 3.0 else "?")
        self._hud_text_small(f"Team: {team_txt}   Status: {alive_txt}", x, y); y += 22

        # Model shape / params / generation
        brain = self._get_brain(aid)
        mlp_str = "unknown"
        pcount = 0
        if brain is not None:
            try:
                shapes = _linear_shape_list(brain)
                mlp_str = _format_mlp_shape(shapes)
                pcount = _param_count(brain)
            except Exception:
                pass
        gen = self._get_generation(aid)
        self._hud_text_small(f"Model: {mlp_str}", x, y); y += 18
        self._hud_text_small(f"Params: {pcount:,}", x, y); y += 18
        self._hud_text_small(f"Gen: {gen if gen is not None else 'n/a'}", x, y); y += 24

        # actions hint
        self._hud_text_small("C = copy brain  |  M = mark", x, y, COLORS["text_dim"]); y += 16
        self._hud_text_small("E = mutate 10% of alive", x, y, COLORS["text_dim"])

    # ---------- bottom HUD ----------
    def _draw_hud(self, stats, cpu_id: torch.Tensor) -> None:
        hud = self._hud_rect()
        pygame.draw.rect(self.screen, COLORS["hud_bg"], hud)

        y0 = hud.y + 6
        self._hud_text(f"Tick {stats.tick} | Elapsed {stats.elapsed_seconds:7.2f}s | Zoom {self.cam.zoom:.2f}x",
                       10, y0)

        # --- compute alive + spawned + by-unit live counts ---
        alive_red = alive_blue = 0
        spawned_red = spawned_blue = 0
        r_s_alive = r_a_alive = b_s_alive = b_a_alive = 0
        if self.registry is not None:
            data = self.registry.agent_data
            alive_mask = (data[:, COL_ALIVE] > 0.5)
            is_red  = (data[:, COL_TEAM] == 2.0)
            is_blue = (data[:, COL_TEAM] == 3.0)
            is_sold = (data[:, COL_UNIT] == 1.0)
            is_arch = (data[:, COL_UNIT] == 2.0)

            alive_red  = int((alive_mask & is_red ).sum().item())
            alive_blue = int((alive_mask & is_blue).sum().item())
            spawned_red  = int(is_red.sum().item())
            spawned_blue = int(is_blue.sum().item())

            r_s_alive = int((alive_mask & is_red  & is_sold).sum().item())
            r_a_alive = int((alive_mask & is_red  & is_arch).sum().item())
            b_s_alive = int((alive_mask & is_blue & is_sold).sum().item())
            b_a_alive = int((alive_mask & is_blue & is_arch).sum().item())

        # --- team lines (with CP and by-unit counts) ---
        self._hud_text(
            f"Red  S:{stats.red.score:7.2f}  CP:{getattr(stats.red, 'cp_points', 0.0):.1f}  "
            f"K:{stats.red.kills}  D:{stats.red.deaths}  "
            f"DMG→:{stats.red.dmg_dealt:.1f}  DMG←:{stats.red.dmg_taken:.1f}  "
            f"Alive:{alive_red}  (S:{r_s_alive} A:{r_a_alive})  Spawned:{spawned_red}",
            10, y0 + 24, (220, 110, 110)
        )
        self._hud_text(
            f"Blue S:{stats.blue.score:7.2f}  CP:{getattr(stats.blue, 'cp_points', 0.0):.1f}  "
            f"K:{stats.blue.kills} D:{stats.blue.deaths}  "
            f"DMG→:{stats.blue.dmg_dealt:.1f} DMG←:{stats.blue.dmg_taken:.1f}  "
            f"Alive:{alive_blue} (S:{b_s_alive} A:{b_a_alive})  Spawned:{spawned_blue}",
            10, y0 + 46, (110, 150, 240)
        )

        # controls row
        self._hud_text_small(
            "Controls: WASD/Arrows pan  |  Wheel zoom  |  C=copy brain  |  M=mark agent  |  E=mutate 10%  |  Esc=quit",
            10, hud.bottom - 20, COLORS["text_dim"]
        )

    # ---------- main loop ----------
    def run(self, engine, registry, stats, tick_limit: int = 0, target_fps: Optional[int] = None) -> None:
        self.registry = registry
        self._build_zone_overlay_cache(engine)  # build once (static over the run)

        fps = int(target_fps or config.TARGET_FPS)
        running = True

        while running:
            running = self._handle_input()

            # one simulation tick
            engine.run_tick()

            # minimal CPU copies (BLOCKING to avoid jitter)
            with torch.no_grad():
                cpu_occ = self.grid[0].detach().cpu().to(torch.int16).numpy()
                cpu_id_t = self.grid[2].detach().cpu().to(torch.int32)  # tensor for markers
                cpu_id_np = cpu_id_t.numpy()

                # Build per-tile unit map on CPU (0 none, 1 soldier, 2 archer)
                unit_np = np.zeros_like(cpu_id_np, dtype=np.int16)
                if self.registry is not None:
                    units_by_id = self.registry.agent_data[:, COL_UNIT].detach().cpu().to(torch.int16).numpy()
                    # safe indexing mask (protect against stale ids)
                    mask = (cpu_id_np >= 0) & (cpu_id_np < units_by_id.shape[0])
                    if mask.any():
                        unit_np[mask] = units_by_id[cpu_id_np[mask]]

            # draw
            self._draw_world(self.screen, cpu_occ, unit_np)
            self._draw_zone_overlays(cpu_occ)  # overlays on top
            self._draw_grid_lines(self.screen)
            self._draw_markers(self.screen, cpu_id_t)
            self._draw_side_panel(cpu_id_t)
            self._draw_hud(stats, cpu_id_t)

            pygame.display.flip()
            self.clock.tick(fps)

            if tick_limit > 0 and stats.tick >= tick_limit:
                running = False

        pygame.quit()
