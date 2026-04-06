import cv2
import numpy as np
import os


# ================== MEASURE ==================
def measure_shakiness(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, prev = cap.read()
    if not ret:
        return None, None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    motions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])

        motion = np.sqrt(dx**2 + dy**2)
        motions.append(motion)

        prev_gray = gray

    cap.release()

    return np.mean(motions), np.std(motions)


# ================== STABILIZE ==================
def stabilize_video(input_path, output_path):
    import cv2
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    import os

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ Cannot open video")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 30

    ret, prev = cap.read()
    if not ret:
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = []

    print("🔍 Estimating motion (PRO)...")

    # ================= PASS 1 =================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            warp = np.eye(2, 3, dtype=np.float32)

            _, warp = cv2.findTransformECC(
                prev_gray, gray, warp,
                cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
            )

            dx = warp[0, 2]
            dy = warp[1, 2]
            da = np.arctan2(warp[1, 0], warp[0, 0])

        except:
            # 🔥 fallback Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            dx = np.mean(flow[..., 0])
            dy = np.mean(flow[..., 1])
            da = 0

        transforms.append([dx, dy, da])
        prev_gray = gray

    cap.release()

    transforms = np.array(transforms)

    # ================= TRAJECTORY =================
    trajectory = np.cumsum(transforms, axis=0)

    # ================= ADAPTIVE SMOOTH =================
    motion_strength = np.mean(np.abs(transforms[:, :2]))

    if motion_strength > 5:
        sigma = 40
        crop_ratio = 0.12
    else:
        sigma = 20
        crop_ratio = 0.06

    print(f"⚙️ Adaptive smoothing sigma={sigma}, crop={crop_ratio}")

    smoothed = np.copy(trajectory)
    for i in range(3):
        smoothed[:, i] = gaussian_filter1d(trajectory[:, i], sigma=sigma)

    diff = smoothed - trajectory
    transforms_smooth = transforms + diff

    # ================= PASS 2 =================
    cap = cv2.VideoCapture(input_path)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    def fix_border(frame):
        h_, w_ = frame.shape[:2]
        crop = frame[
            int(crop_ratio*h_):int((1-crop_ratio)*h_),
            int(crop_ratio*w_):int((1-crop_ratio)*w_)
        ]
        return cv2.resize(crop, (w_, h_))

    print("🎥 Stabilizing PRO...")

    i = 0

    while True:
        ret, frame = cap.read()
        if not ret or i >= len(transforms_smooth):
            break

        dx, dy, da = transforms_smooth[i]

        # 🔥 hạn chế over warp
        dx = np.clip(dx, -50, 50)
        dy = np.clip(dy, -50, 50)
        da = np.clip(da, -0.3, 0.3)

        center = (w // 2, h // 2)

        m = cv2.getRotationMatrix2D(center, np.degrees(da), 1)
        m[0, 2] += dx
        m[1, 2] += dy

        stabilized = cv2.warpAffine(
            frame,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        stabilized = fix_border(stabilized)

        out.write(stabilized)
        i += 1

    cap.release()
    out.release()

    print("✅ DONE PRO:", output_path)

    # ================= ENCODE =================
    fixed_output = output_path.replace(".mp4", "_fixed.mp4")

    os.system(
        f'ffmpeg -y -i "{output_path}" -vcodec libx264 -crf 18 -preset slow -pix_fmt yuv420p "{fixed_output}"'
    )

    if os.path.exists(fixed_output):
        print("🎬 FINAL:", fixed_output)
        return fixed_output

    return output_path
def create_comparison_video(input_path, stabilized_path, output_path):
    cap1 = cv2.VideoCapture(input_path)
    cap2 = cv2.VideoCapture(stabilized_path)

    if not cap1.isOpened() or not cap2.isOpened():
        return None

    w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap1.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 30

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w * 2, h)
    )

    frame_count = 0

    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()

        if not ret1 or not ret2:
            break

        f1 = cv2.resize(f1, (w, h))
        f2 = cv2.resize(f2, (w, h))

        cv2.putText(f1, "Before", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(f2, "After", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined = np.hstack((f1, f2))

        # line giữa
        cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)

        out.write(combined)
        frame_count += 1

    cap1.release()
    cap2.release()
    out.release()

    if frame_count == 0:
        return None

    # encode web
    fixed_output = output_path.replace(".mp4", "_fixed.mp4")

    os.system(
        f'ffmpeg -y -i "{output_path}" -vcodec libx264 -crf 23 -preset fast -pix_fmt yuv420p "{fixed_output}"'
    )

    if os.path.exists(fixed_output):
        return fixed_output

    return output_path