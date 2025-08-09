import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# ===== Optional: onnxruntime presence =====
try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False


# ---------------- ONNX Detector (robust) ----------------
class ONNXFaceDetector:
    def __init__(self, onnx_path,
                 input_size=(640, 640),
                 conf_thres=0.25,
                 iou_thres=0.45,
                 use_invert=False,
                 clahe_clip=2.0, clahe_grid=8,
                 providers=None):

        if onnx_path is None or not str(onnx_path):
            raise ValueError("ONNX 모델 경로가 비어 있습니다.")

        if not HAS_ORT:
            raise RuntimeError("onnxruntime이 설치되지 않았습니다. pip install onnxruntime 또는 onnxruntime-gpu")

        self.onnx_path = onnx_path
        self.input_size = input_size  # (w, h)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.use_invert = use_invert
        self.clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(clahe_grid, clahe_grid))

        ort_ver = getattr(ort, "__version__", "unknown")
        avail = set(ort.get_available_providers())

        req = providers or ["CPUExecutionProvider"]
        req = list(req)

        # req ∩ avail, 없으면 CPU 폴백
        chosen = [p for p in req if p in avail]
        if not chosen:
            if "CPUExecutionProvider" in avail:
                chosen = ["CPUExecutionProvider"]
            else:
                raise RuntimeError(f"사용 가능한 ExecutionProvider가 없습니다. available={list(avail)}")

        # SessionOptions (로그 레벨 낮춤)
        so = ort.SessionOptions()
        # 0=verbose, 1=info, 2=warning, 3=error, 4=fatal
        so.log_severity_level = 2

        try:
            self.sess = ort.InferenceSession(self.onnx_path, sess_options=so, providers=chosen)
        except TypeError as e:
            # 구버전 onnxruntime은 providers 인자 미지원
            raise RuntimeError(
                f"InferenceSession 생성 실패: onnxruntime {ort_ver}. "
                f"onnxruntime(또는 onnxruntime-gpu)을 최신으로 업데이트하세요."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "ONNX 세션 생성 실패\n"
                f"- ort version: {ort_ver}\n"
                f"- requested providers: {req}\n"
                f"- chosen providers: {chosen}\n"
                f"- available providers: {list(avail)}\n"
                f"- model: {self.onnx_path}\n"
                f"원인: {e}"
            ) from e

        try:
            self.inp_name = self.sess.get_inputs()[0].name
        except Exception as e:
            raise RuntimeError(f"모델 입력 정보를 읽는 중 오류: {e}")

    def set_params(self, input_w, input_h, conf, iou, clip, inv):
        self.input_size = (int(input_w), int(input_h))
        self.conf_thres = float(conf)
        self.iou_thres = float(iou)
        self.clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
        self.use_invert = bool(int(inv))

    def _clahe_proc(self, gray):
        return self.clahe.apply(gray)

    @staticmethod
    def _nms(boxes, scores, iou_thres):
        # boxes: [N,4] (x1,y1,x2,y2), scores: [N]
        if boxes.size == 0:
            return []
        idxs = scores.argsort()[::-1]
        keep = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            if idxs.size == 1:
                break
            rest = idxs[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-6)
            idxs = rest[iou < iou_thres]
        return keep

    def _preprocess(self, img_bgr_or_gray):
        # 입력: 그레이 또는 BGR, 출력: (1,3,H,W) float32 [0..1], scale_to=(W,H)
        if len(img_bgr_or_gray.shape) == 2:
            inp = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
        else:
            inp = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)

        in_w, in_h = self.input_size
        resized = cv2.resize(inp, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]  # NCHW
        return blob, (inp.shape[1], inp.shape[0])  # (W,H)

    def _parse_outputs(self, ort_out, scale_from, scale_to):
        """
        ort_out[0]가 [N,5] 또는 [1,N,5] (x1,y1,x2,y2,conf) 형식이라고 가정
        scale_from: 네트 입력 사이즈 (W_in, H_in)
        scale_to:   표시할 이미지 사이즈 (W, H)
        """
        preds = ort_out[0]
        if preds.ndim == 3:  # (1,N,5)
            preds = preds[0]
        if preds.size == 0:
            return np.zeros((0, 4), dtype=np.int32), np.array([])

        # conf filter
        conf = preds[:, 4]
        m = conf >= self.conf_thres
        preds = preds[m]
        if preds.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.int32), np.array([])

        # scale back
        in_w, in_h = scale_from
        W, H = scale_to
        sx, sy = W / in_w, H / in_h

        boxes = preds[:, :4].copy()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        boxes = boxes.clip(min=0)
        scores = preds[:, 4]

        keep = self._nms(boxes, scores, self.iou_thres)
        return boxes[keep].astype(np.int32), scores[keep]

def detect(self, rgb_img):
    # normalize size
    h, w = rgb_img.shape[:2]
    scale = 800.0 / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(rgb_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        rgb = rgb_img.copy()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # preprocessed variants
    g_plain = gray
    g_clahe = self._clahe(gray)             # CLAHE 강도는 self.clipLimit 슬라이더로 반영됨
    g_inv = cv2.bitwise_not(gray)
    g_clahe_inv = cv2.bitwise_not(g_clahe)

    # 탐지 시도 시나리오(여러 후보 중 먼저 잡히는 걸로 박스만 얻음)
    scenarios = (
        [("haar", g_inv), ("haar", g_clahe_inv), ("lbp", g_inv), ("lbp", g_clahe_inv)]
        if self.use_invert
        else [("haar", g_plain), ("haar", g_clahe), ("lbp", g_plain), ("lbp", g_clahe)]
    )

    # === 표시 정책: "항상" 현재 설정 기반 전처리 배경을 먼저 깐다 ===
    # invert 토글 기준으로 어떤 전처리 이미지를 보여줄지 결정
    display_gray = g_clahe_inv if self.use_invert else g_clahe
    # show_processed가 True면 전처리 이미지를 컬러로 변환해 배경으로, 아니면 원본 컬러
    result = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2RGB) if self.show_processed else rgb.copy()

    # === 탐지: 박스만 현재 result 위에 그린다 (result를 덮어쓰지 말 것!) ===
    best_faces = []
    for kind, g in scenarios:
        classifier = self.haar if kind == "haar" else self.lbp
        faces = self._detect_once(g, classifier)
        if len(faces) > 0:
            best_faces = faces
            break

    # draw boxes (있으면)
    for (x, y, wf, hf) in best_faces:
        cv2.rectangle(result, (x, y), (x + wf, y + hf), (0, 255, 0), 2)

    return result, best_faces


# ---------------- Haar/LBP Detector ----------------
class HaarFaceDetector:
    def __init__(self, show_processed=True):
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.lbp = cv2.CascadeClassifier(cv2.data.haarcascades + "lbpcascade_frontalface.xml")
        self.scaleFactor = 1.15
        self.minNeighbors = 5
        self.minSize = 40
        self.clipLimit = 2.0
        self.use_invert = False
        self.show_processed = show_processed  # 전처리 영상 표시 여부

    def set_params(self, sf, mn, ms, clip, inv):
        self.scaleFactor = float(sf)
        self.minNeighbors = int(mn)
        self.minSize = int(ms)
        self.clipLimit = float(clip)
        self.use_invert = bool(int(inv))

    def _clahe(self, gray):
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _detect_once(self, gray, classifier):
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=(self.minSize, self.minSize),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def detect(self, rgb_img):
        # normalize size
        h, w = rgb_img.shape[:2]
        scale = 800.0 / max(h, w)
        if scale < 1.0:
            rgb = cv2.resize(rgb_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            rgb = rgb_img.copy()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # scenarios
        g_plain = gray
        g_clahe = self._clahe(gray)
        g_inv = cv2.bitwise_not(gray)
        g_clahe_inv = cv2.bitwise_not(g_clahe)

        if self.use_invert:
            scenarios = [("haar", g_inv), ("haar", g_clahe_inv), ("lbp", g_inv), ("lbp", g_clahe_inv)]
        else:
            scenarios = [("haar", g_plain), ("haar", g_clahe), ("lbp", g_plain), ("lbp", g_clahe)]

        # 기본은 원본 컬러, show_processed면 탐지에 사용한 그레이 영상을 컬러로 변환해 표시
        result = rgb.copy()
        for kind, g in scenarios:
            classifier = self.haar if kind == "haar" else self.lbp
            faces = self._detect_once(g, classifier)
            if len(faces) > 0:
                if self.show_processed:
                    result = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
                for (x, y, wf, hf) in faces:
                    cv2.rectangle(result, (x, y), (x + wf, y + hf), (0, 255, 0), 2)
                return result, faces
        return result, []


# ---------------- Unified GUI ----------------
class FaceDetectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection (Haar / ONNX)")
        self.original_image = None
        self.result_image = None
        self.backend = tk.StringVar(value="Haar")  # "Haar" or "ONNX"
        self.onnx_model_path = None
        self.onnx_detector = None
        self.haar_detector = HaarFaceDetector(show_processed=True)

        # Top: backend selector
        top = tk.Frame(root); top.pack(pady=6)
        tk.Label(top, text="Backend:").pack(side=tk.LEFT)
        self.backend_menu = tk.OptionMenu(top, self.backend, "Haar", "ONNX")
        self.backend_menu.pack(side=tk.LEFT, padx=8)

        # trace로 변경 (OptionMenu의 command 인자 안 씀)
        self.backend.trace_add("write", lambda *args: self._on_backend_change(self.backend.get()))

        self.btn_load_model = tk.Button(top, text="Load ONNX Model", command=self._choose_onnx, state=tk.DISABLED)
        self.btn_load_model.pack(side=tk.LEFT, padx=5)

        self.provider_var = tk.StringVar(value="CPUExecutionProvider")
        self.provider_menu = tk.OptionMenu(top, self.provider_var, "CPUExecutionProvider", "CUDAExecutionProvider")
        self.provider_menu.config(state=tk.DISABLED)
        self.provider_menu.pack(side=tk.LEFT, padx=5)

        # Show processed checkbox (Haar)
        self.show_proc_var = tk.IntVar(value=1)
        self.chk_show_proc = tk.Checkbutton(top, text="Show processed view (Haar)", variable=self.show_proc_var,
                                            command=self._toggle_show_processed)
        self.chk_show_proc.pack(side=tk.LEFT, padx=8)

        # Image panels
        self.image_frame = tk.Frame(root)
        self.image_frame.pack()
        self.left = tk.Label(self.image_frame, text="원본"); self.left.pack(side=tk.LEFT, padx=5, pady=5)
        self.right = tk.Label(self.image_frame, text="결과"); self.right.pack(side=tk.RIGHT, padx=5, pady=5)

        # Buttons
        self.btn_frame = tk.Frame(root); self.btn_frame.pack(pady=6)
        tk.Button(self.btn_frame, text="Load", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Run", command=self.run_detection).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Save", command=self.save_image).pack(side=tk.LEFT, padx=5)

        # Param panels container
        self.param_container = tk.Frame(root); self.param_container.pack(pady=8, fill=tk.X)

        # Haar panel
        self.haar_panel = tk.LabelFrame(self.param_container, text="Haar/LBP Params")
        self._build_haar_panel(self.haar_panel)

        # ONNX panel
        self.onnx_panel = tk.LabelFrame(self.param_container, text="ONNX Params")
        self._build_onnx_panel(self.onnx_panel)

        # initial layout
        self.haar_panel.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)
        # ONNX panel은 기본 숨김

        # auto-run hookup (안전: *args)
        self._hook_autorun(self.haar_sliders + self.common_sliders)

        # 초기 백엔드 상태 반영
        self._on_backend_change("Haar")

    def _toggle_show_processed(self):
        self.haar_detector.show_processed = bool(self.show_proc_var.get())
        self.run_detection()

    # ---------- UI builders ----------
    def _build_haar_panel(self, parent):
        self.h_sf = self._labeled_scale(parent, "scaleFactor (1.05~1.5)", 1.05, 1.50, 0.01, 1.15, row=0)
        self.h_mn = self._labeled_scale(parent, "minNeighbors (1~12)", 1, 12, 1, 5, row=1)
        self.h_ms = self._labeled_scale(parent, "minSize px (20~200)", 20, 200, 5, 40, row=2)
        self.h_clip = self._labeled_scale(parent, "CLAHE clipLimit (1.0~5.0)", 1.0, 5.0, 0.1, 2.0, row=3)
        self.h_inv = self._labeled_scale(parent, "Invert(반전) 사용", 0, 1, 1, 0, row=4, length=160)
        self.haar_sliders = [self.h_sf, self.h_mn, self.h_ms, self.h_clip, self.h_inv]

        # 공통(현재는 없음, 확장 대비)
        self.common_sliders = []

    def _build_onnx_panel(self, parent):
        r = 0
        self.in_w = self._labeled_scale(parent, "Input Width (320~1280)", 320, 1280, 32, 640, row=r); r += 1
        self.in_h = self._labeled_scale(parent, "Input Height (320~1280)", 320, 1280, 32, 640, row=r); r += 1
        self.o_conf = self._labeled_scale(parent, "Conf Thres (0.05~0.9)", 0.05, 0.90, 0.01, 0.25, row=r); r += 1
        self.o_iou = self._labeled_scale(parent, "IoU Thres (0.10~0.9)", 0.10, 0.90, 0.01, 0.45, row=r); r += 1
        self.o_clip = self._labeled_scale(parent, "CLAHE clipLimit (1.0~5.0)", 1.0, 5.0, 0.1, 2.0, row=r); r += 1
        self.o_inv = self._labeled_scale(parent, "Invert(반전) 사용", 0, 1, 1, 0, row=r); r += 1
        self.onnx_sliders = [self.in_w, self.in_h, self.o_conf, self.o_iou, self.o_clip, self.o_inv]

    def _labeled_scale(self, parent, text, minv, maxv, res, default, row=0, length=240):
        tk.Label(parent, text=text).grid(row=row, column=0, sticky="w")
        s = tk.Scale(parent, from_=minv, to=maxv, resolution=res, orient=tk.HORIZONTAL, length=length)
        s.set(default); s.grid(row=row, column=1)
        return s

    def _hook_autorun(self, widgets):
        for s in widgets:
            s.configure(command=lambda *args: self.run_detection())

    # ---------- Backend switch / model ----------
    def _on_backend_change(self, value):
        if value == "ONNX":
            self.btn_load_model.config(state=tk.NORMAL)
            self.provider_menu.config(state=(tk.NORMAL if HAS_ORT else tk.DISABLED))
            # show ONNX panel / hide Haar panel
            self.onnx_panel.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)
            self.haar_panel.pack_forget()
            # autorun for onnx knobs
            self._hook_autorun(self.onnx_sliders)
        else:
            self.btn_load_model.config(state=tk.DISABLED)
            self.provider_menu.config(state=tk.DISABLED)
            # show Haar / hide ONNX
            self.haar_panel.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)
            self.onnx_panel.pack_forget()
            # autorun for haar knobs
            self._hook_autorun(self.haar_sliders + self.common_sliders)

        # 즉시 한 번 갱신
        self.run_detection()

    def _choose_onnx(self):
        if not HAS_ORT:
            messagebox.showerror("onnxruntime missing",
                                 "pip install onnxruntime 또는 onnxruntime-gpu 로 설치하세요.")
            return
        path = filedialog.askopenfilename(filetypes=[("ONNX model", "*.onnx")])
        if not path:
            return
        self.onnx_model_path = path
        try:
            req = [self.provider_var.get()]
            avail = ort.get_available_providers()
            if req[0] not in avail:
                messagebox.showwarning(
                    "Provider fallback",
                    f"{req[0]} 사용 불가. 사용 가능: {avail}\nCPUExecutionProvider로 폴백합니다."
                )
                req = ["CPUExecutionProvider"]

            self.onnx_detector = ONNXFaceDetector(path, providers=req)
            messagebox.showinfo("ONNX", f"모델 로드 완료:\n{path}\nProvider: {req[0]}")
        except Exception as e:
            self.onnx_detector = None
            messagebox.showerror("ONNX Load Error", str(e))

    # ---------- IO ----------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Load Error", "이미지를 읽을 수 없습니다.")
            return
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.original_image = img
        self._show(self.left, img)
        self.run_detection()

    def save_image(self):
        if self.result_image is None:
            messagebox.showwarning("Save", "저장할 결과 이미지가 없습니다.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if not path:
            return
        cv2.imwrite(path, cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Save", f"저장 완료: {path}")

    # ---------- Inference ----------
    def run_detection(self):
        if self.original_image is None:
            return
        backend = self.backend.get()
        if backend == "Haar":
            self.haar_detector.set_params(
                self.h_sf.get(), self.h_mn.get(), self.h_ms.get(),
                self.h_clip.get(), self.h_inv.get()
            )
            out, faces = self.haar_detector.detect(self.original_image)
        else:
            if self.onnx_detector is None:
                if self.onnx_model_path is None:
                    messagebox.showwarning("ONNX", "먼저 ONNX 모델을 로드하세요.")
                    return
                # 모델 경로는 있는데 detector가 None인 경우(초기화 지연)
                try:
                    req = [self.provider_var.get()]
                    if HAS_ORT and req[0] not in ort.get_available_providers():
                        req = ["CPUExecutionProvider"]
                    self.onnx_detector = ONNXFaceDetector(self.onnx_model_path, providers=req)
                except Exception as e:
                    messagebox.showerror("ONNX Init Error", str(e))
                    return
            try:
                self.onnx_detector.set_params(
                    self.in_w.get(), self.in_h.get(), self.o_conf.get(),
                    self.o_iou.get(), self.o_clip.get(), self.o_inv.get()
                )
                out, faces = self.onnx_detector.detect(self.original_image)
            except Exception as e:
                messagebox.showerror("ONNX Inference Error", str(e))
                return

        self.result_image = out
        self._show(self.right, out)

    def _show(self, label, img):
        im = Image.fromarray(img)
        im.thumbnail((500, 500))
        tkimg = ImageTk.PhotoImage(im)
        label.configure(image=tkimg)
        label.image = tkimg


# ---------------- Main ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectApp(root)
    root.mainloop()
