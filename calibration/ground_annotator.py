# -*- coding: utf-8 -*-
"""
地面标注器：选择像素点并输入对应球场坐标，输出标注图片和 JSON。

用法：
  python -m calibration.ground_annotator --image path/to/ground_636.png
"""

import argparse
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

_MIN_GROUND_POINTS = 6


class AnnotatorApp(tk.Tk):
    def __init__(self, world_points: list[tuple[float, float, float]] | None = None):
        super().__init__()
        self.title("地面标注器")
        self.geometry("1400x800")
        self.world_points = world_points or []

        # --- 状态 ---
        self.img_path: Path | None = None
        self.cv_img: np.ndarray | None = None
        self.pil_img: Image.Image | None = None
        self.annotations: list[
            dict[str, tuple[float, float, float] | tuple[int, int]]
        ] = []
        self.annotating = False

        # 视图变换
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._drag_start = None

        # 缩放图片缓存
        self._cached_scale = None
        self._cached_tk_img = None

        # 缩放防抖
        self._zoom_timer = None
        self._zoom_anchor = None  # 缩放时鼠标锚点

        # Canvas 上的标注元素 id：[(circle_id, text_id), ...]
        self._marker_ids: list[tuple[int, int]] = []

        # --- 工具栏 ---
        toolbar = tk.Frame(self, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = tk.Button(toolbar, text="打开图片", command=self._open_image)
        self.btn_open.pack(side=tk.LEFT, padx=4, pady=2)

        self.btn_annotate = tk.Button(toolbar, text="开始标注", command=self._toggle_annotate)
        self.btn_annotate.pack(side=tk.LEFT, padx=4, pady=2)

        self.btn_delete = tk.Button(toolbar, text="删除标注", command=self._delete_last)
        self.btn_delete.pack(side=tk.LEFT, padx=4, pady=2)

        self.btn_save = tk.Button(toolbar, text="保存", command=self._save)
        self.btn_save.pack(side=tk.LEFT, padx=4, pady=2)

        self.status_label = tk.Label(toolbar, text="请打开图片", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=12)

        # --- 主区域：左侧点列表 + 右侧 Canvas ---
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧：标注点列表面板
        self.panel = tk.Frame(main_frame, width=220, bd=1, relief=tk.SUNKEN)
        self.panel.pack(side=tk.LEFT, fill=tk.Y)
        self.panel.pack_propagate(False)

        tk.Label(self.panel, text="标注点列表", font=("Arial", 11, "bold")).pack(
            pady=(8, 4))
        self.count_label = tk.Label(
            self.panel,
            text=self._count_text(),
            fg="#666",
        )
        self.count_label.pack()
        self.point_list = tk.Frame(self.panel)
        self.point_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 右侧：Canvas
        self.canvas = tk.Canvas(main_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._img_on_canvas = None

        # --- 事件绑定 ---
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<ButtonPress-2>", self._on_drag_start)
        self.canvas.bind("<B2-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonPress-3>", self._on_drag_start)
        self.canvas.bind("<B3-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonPress-1>", self._on_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self._resize_pending = False

    # ----------------------------------------------------------------
    # 坐标转换
    # ----------------------------------------------------------------
    def _canvas_to_pixel(self, cx, cy):
        px = (cx - self.offset_x) / self.scale
        py = (cy - self.offset_y) / self.scale
        return px, py

    def _pixel_to_canvas(self, px, py):
        cx = px * self.scale + self.offset_x
        cy = py * self.scale + self.offset_y
        return cx, cy

    # ----------------------------------------------------------------
    # 左侧面板更新
    # ----------------------------------------------------------------
    def _required_points(self):
        return len(self.world_points) if self.world_points else _MIN_GROUND_POINTS

    def _count_text(self):
        return f"已标注 {len(self.annotations)}/{self._required_points()} 个点"

    def _update_panel(self):
        for child in self.point_list.winfo_children():
            child.destroy()
        self.count_label.config(text=self._count_text())
        for i, item in enumerate(self.annotations, start=1):
            x, y, z = item["world"]
            px, py = item["pixel"]
            tk.Label(
                self.point_list,
                text=f"#{i} W=({x:g}, {y:g}, {z:g}) P=({px}, {py})",
                anchor=tk.W,
                font=("Consolas", 10),
                fg="#00aa00",
            ).pack(fill=tk.X, pady=2)

    # ----------------------------------------------------------------
    # 图片显示
    # ----------------------------------------------------------------
    def _open_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        if not path:
            return
        self._load_image(Path(path))

    def _load_image(self, path: Path):
        self.img_path = Path(path)
        self.cv_img = cv2.imread(str(self.img_path), cv2.IMREAD_COLOR)
        if self.cv_img is None:
            messagebox.showerror("错误", f"无法读取图片:\n{self.img_path}")
            return
        self.pil_img = Image.fromarray(cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB))

        # 重置
        self.annotations.clear()
        self._marker_ids.clear()
        self.annotating = False
        self._cached_scale = None
        self._cached_tk_img = None
        self.btn_annotate.config(text="开始标注", relief=tk.RAISED)
        self.canvas.config(cursor="")

        self._fit_image_to_canvas()

        self._update_panel()
        self.title(f"地面标注器 - {self.img_path.parent.name}/{self.img_path.name}")
        if self.world_points:
            point = self.world_points[0]
            self.status_label.config(
                text=f"第 1 个点: ({point[0]:g}, {point[1]:g}, {point[2]:g})"
            )
        else:
            iw, ih = self.pil_img.size
            self.status_label.config(text=f"{self.img_path.name}  ({iw}x{ih})")

    def _fit_image_to_canvas(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih = self.pil_img.size
        self.scale = min(cw / iw, ch / ih, 1.0)
        self.offset_x = (cw - iw * self.scale) / 2
        self.offset_y = (ch - ih * self.scale) / 2
        self._cached_scale = None
        self._full_redraw()

    def _render_scaled_image(self):
        """生成缩放后的 PhotoImage（仅在 scale 变化时调用）"""
        iw, ih = self.pil_img.size
        new_w = max(1, int(iw * self.scale))
        new_h = max(1, int(ih * self.scale))
        resized = self.pil_img.resize((new_w, new_h), Image.BILINEAR)
        self._cached_tk_img = ImageTk.PhotoImage(resized)
        self._cached_scale = self.scale

    def _full_redraw(self):
        """完整重绘（缩放变化、窗口 resize 时调用）"""
        self.canvas.delete("all")
        self._marker_ids.clear()

        if self.pil_img is None:
            return

        if self._cached_scale != self.scale or self._cached_tk_img is None:
            self._render_scaled_image()

        self._img_on_canvas = self.canvas.create_image(
            self.offset_x, self.offset_y, anchor=tk.NW, image=self._cached_tk_img
        )

        for i, item in enumerate(self.annotations):
            px, py = item["pixel"]
            self._draw_marker(i + 1, px, py)

    def _draw_marker(self, number, px, py):
        cx, cy = self._pixel_to_canvas(px, py)
        r = max(10, int(12 * min(self.scale, 2.0)))
        font_size = max(8, int(10 * min(self.scale, 2.0)))

        cid = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill="#ff4444", outline="white", width=2
        )
        tid = self.canvas.create_text(
            cx, cy, text=str(number),
            fill="white", font=("Arial", font_size, "bold")
        )
        self._marker_ids.append((cid, tid))

    # ----------------------------------------------------------------
    # 缩放 & 拖动
    # ----------------------------------------------------------------
    def _on_scroll(self, event):
        if self.pil_img is None:
            return

        # 计算新 scale 和 offset
        cx, cy = event.x, event.y
        old_scale = self.scale
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.scale = max(0.05, min(50.0, self.scale * factor))
        ratio = self.scale / old_scale

        self.offset_x = cx - (cx - self.offset_x) * ratio
        self.offset_y = cy - (cy - self.offset_y) * ratio

        # 即时预览：用 canvas.scale() 缩放现有元素（无需重新生成图片）
        self.canvas.scale("all", cx, cy, ratio, ratio)

        # 防抖：连续滚轮事件合并，停止 80ms 后才真正 resize 图片
        if self._zoom_timer is not None:
            self.after_cancel(self._zoom_timer)
        self._zoom_timer = self.after(80, self._deferred_zoom)

    def _deferred_zoom(self):
        """滚轮停止后，生成精确的缩放图片"""
        self._zoom_timer = None
        self._full_redraw()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag_move(self, event):
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self.offset_x += dx
        self.offset_y += dy
        self.canvas.move("all", dx, dy)

    def _on_canvas_resize(self, event):
        if self.pil_img is None:
            return
        if not self._resize_pending:
            self._resize_pending = True
            self.after(100, self._deferred_resize)

    def _deferred_resize(self):
        self._resize_pending = False
        self._fit_image_to_canvas()

    # ----------------------------------------------------------------
    # 标注
    # ----------------------------------------------------------------
    def _toggle_annotate(self):
        if self.world_points and len(self.annotations) == len(self.world_points):
            messagebox.showinfo("提示", "预设点已全部标注，请直接保存")
            return
        self.annotating = not self.annotating
        if self.annotating:
            self.btn_annotate.config(text="停止标注", relief=tk.SUNKEN)
            self.canvas.config(cursor="crosshair")
        else:
            self.btn_annotate.config(text="开始标注", relief=tk.RAISED)
            self.canvas.config(cursor="")

    def _on_click(self, event):
        if not self.annotating or self.pil_img is None:
            return

        px, py = self._canvas_to_pixel(event.x, event.y)
        iw, ih = self.pil_img.size
        if px < 0 or py < 0 or px >= iw or py >= ih:
            return
        px_int, py_int = int(round(px)), int(round(py))
        if self.world_points:
            world = self.world_points[len(self.annotations)]
        else:
            world = self._ask_world_point(px_int, py_int)
        if world is None:
            return
        self.annotations.append({"pixel": (px_int, py_int), "world": world})
        n = len(self.annotations)
        self._draw_marker(n, px_int, py_int)

        if self.world_points and n == len(self.world_points):
            self.annotating = False
            self.btn_annotate.config(text="开始标注", relief=tk.RAISED)
            self.canvas.config(cursor="")
            self.status_label.config(text="预设点已全部标注，请检查后保存")
        elif self.world_points:
            next_point = self.world_points[n]
            self.status_label.config(
                text=f"下一个点: ({next_point[0]:g}, {next_point[1]:g}, {next_point[2]:g})"
            )
        else:
            self.status_label.config(
                text=f"标注 #{n}: pixel=({px_int}, {py_int})  "
                     f"world=({world[0]:g}, {world[1]:g}, {world[2]:g})")
        self._update_panel()

    def _ask_world_point(self, px: int, py: int) -> tuple[float, float, float] | None:
        while True:
            raw = simpledialog.askstring(
                "输入球场坐标",
                f"像素点 ({px}, {py}) 对应的球场坐标，单位米：\n"
                "格式: x,y,z   例如: 0,6,0",
                parent=self,
            )
            if raw is None:
                return None
            parts = raw.replace("，", ",").split(",")
            if len(parts) != 3:
                messagebox.showwarning("格式错误", "请输入 3 个数字，例如: 0,6,0")
                continue
            try:
                return tuple(float(value.strip()) for value in parts)
            except ValueError:
                messagebox.showwarning("格式错误", "坐标必须是数字，例如: 0,6,0")

    def _delete_last(self):
        if not self.annotations:
            messagebox.showinfo("提示", "没有标注可以删除")
            return
        self.annotations.pop()
        if self._marker_ids:
            cid, tid = self._marker_ids.pop()
            self.canvas.delete(cid)
            self.canvas.delete(tid)
        n = len(self.annotations)
        self.status_label.config(text=f"已删除，剩余 {n} 个标注")
        self._update_panel()

    # ----------------------------------------------------------------
    # 保存
    # ----------------------------------------------------------------
    def _save(self):
        if self.cv_img is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
        required_points = self._required_points()
        if len(self.annotations) < required_points:
            messagebox.showwarning(
                "提示",
                f"需要 {required_points} 个点，"
                f"当前已标注 {len(self.annotations)} 个",
            )
            return

        stem = self.img_path.stem
        out_dir = self.img_path.parent

        # --- 生成标注图片 ---
        annotated = self.cv_img.copy()
        for i, item in enumerate(self.annotations):
            px, py = item["pixel"]
            number = i + 1
            label = str(number)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            r = max(tw, th) // 2 + 8

            cv2.circle(annotated, (px, py), r, (0, 0, 255), -1)
            cv2.circle(annotated, (px, py), r, (255, 255, 255), 2)
            tx = px - tw // 2
            ty = py + th // 2
            cv2.putText(annotated, label, (tx, ty), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

        img_out = out_dir / f"{stem}_annotated.png"
        cv2.imwrite(str(img_out), annotated)

        # --- 生成 JSON ---
        data = {}
        for i, item in enumerate(self.annotations):
            x, y, z = item["world"]
            px, py = item["pixel"]
            data[str(i + 1)] = [[x, y, z], [px, py]]

        json_out = out_dir / f"{stem}_annotations.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        self.status_label.config(text=f"已保存: {img_out.name}, {json_out.name}")
        messagebox.showinfo("保存成功",
                            f"标注图片: {img_out}\nJSON: {json_out}")


def main():
    parser = argparse.ArgumentParser(description="地面坐标点标注器")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument(
        "--point", type=float, nargs=3, action="append", default=None,
        metavar=("X", "Y", "Z"),
    )
    args = parser.parse_args()

    world_points = [tuple(point) for point in args.point] if args.point else None
    app = AnnotatorApp(world_points)
    if args.image is not None:
        app._load_image(args.image.resolve())
    app.mainloop()


if __name__ == "__main__":
    main()
