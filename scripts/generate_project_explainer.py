"""Create the illustrated portfolio project explainer PDF with ReportLab."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.colors import HexColor, Color
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas


PAGE_W, PAGE_H = A4
MARGIN = 42
CONTENT_W = PAGE_W - 2 * MARGIN

BG = HexColor("#0B1220")
PANEL = HexColor("#141E30")
PANEL_ALT = HexColor("#18253A")
WHITE = HexColor("#F3F6FB")
INK = HexColor("#162235")
MUTED = HexColor("#A9B6C8")
SUBTLE = HexColor("#718096")
BLUE = HexColor("#58A6FF")
GREEN = HexColor("#57D6A3")
ORANGE = HexColor("#FFB86C")
RED = HexColor("#FF7B72")
PURPLE = HexColor("#C39BFF")
GRID = HexColor("#2B3850")


def _text(c: canvas.Canvas, x: float, y: float, value: str, size: float = 10,
          color: Color = WHITE, font: str = "Helvetica") -> None:
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(x, y, value)


def _right_text(c: canvas.Canvas, x: float, y: float, value: str, size: float = 10,
                color: Color = WHITE, font: str = "Helvetica") -> None:
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawRightString(x, y, value)


def _paragraph(c: canvas.Canvas, x: float, y: float, value: str, width: float,
               size: float = 10, leading: float | None = None,
               color: Color = WHITE, font: str = "Helvetica") -> float:
    line_height = leading or size * 1.38
    c.setFillColor(color)
    c.setFont(font, size)
    for line in simpleSplit(value, font, size, width):
        c.drawString(x, y, line)
        y -= line_height
    return y


def _rounded(c: canvas.Canvas, x: float, y: float, w: float, h: float,
             fill: Color = PANEL, stroke: Color | None = None, radius: float = 12) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1 if stroke else 0)


def _header(c: canvas.Canvas, page: int, section: str) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    _text(c, MARGIN, PAGE_H - 30, "VISUAL SERVO MUJOCO", 8, GREEN, "Helvetica-Bold")
    _right_text(c, PAGE_W - MARGIN, PAGE_H - 30, section.upper(), 8, MUTED, "Helvetica-Bold")
    c.setStrokeColor(GRID)
    c.setLineWidth(0.6)
    c.line(MARGIN, PAGE_H - 40, PAGE_W - MARGIN, PAGE_H - 40)
    c.line(MARGIN, 31, PAGE_W - MARGIN, 31)
    _text(c, MARGIN, 18, "Portfolio project explainer | camera and simulation evidence, not robot performance", 7.5, MUTED)
    _right_text(c, PAGE_W - MARGIN, 18, f"{page:02d}", 8, MUTED, "Helvetica-Bold")


def _title(c: canvas.Canvas, title: str, subtitle: str, top: float = PAGE_H - 73) -> float:
    title_width = c.stringWidth(title, "Helvetica-Bold", 23)
    title_size = min(23, 23 * CONTENT_W / title_width)
    _text(c, MARGIN, top, title, title_size, WHITE, "Helvetica-Bold")
    y = _paragraph(c, MARGIN, top - 22, subtitle, CONTENT_W, 9.5, 13, MUTED)
    return y - 12


def _section_label(c: canvas.Canvas, x: float, y: float, label: str,
                   color: Color = GREEN) -> None:
    _text(c, x, y, label.upper(), 8, color, "Helvetica-Bold")


def _table(c: canvas.Canvas, x: float, top: float, widths: list[float],
           rows: list[list[str]], *, font_size: float = 8.2,
           header: bool = True, leading: float | None = None) -> float:
    line_height = leading or font_size * 1.2
    y = top
    for row_index, row in enumerate(rows):
        row_lines = [
            simpleSplit(value, "Helvetica-Bold" if row_index == 0 and header else "Helvetica",
                        font_size, col_width - 12)
            for value, col_width in zip(row, widths)
        ]
        height = max(23.0, max(len(lines) for lines in row_lines) * line_height + 11)
        c.setFillColor(PANEL_ALT if row_index % 2 else PANEL)
        if row_index == 0 and header:
            c.setFillColor(HexColor("#263750"))
        c.rect(x, y - height, sum(widths), height, fill=1, stroke=0)
        cursor_x = x
        for col_index, (value, col_width) in enumerate(zip(row, widths)):
            font = "Helvetica-Bold" if row_index == 0 and header else "Helvetica"
            color = GREEN if row_index == 0 and header else WHITE
            c.setFillColor(color)
            c.setFont(font, font_size)
            for line_index, line in enumerate(row_lines[col_index]):
                c.drawString(cursor_x + 6, y - 9 - line_index * line_height, line)
            if col_index:
                c.setStrokeColor(GRID)
                c.setLineWidth(0.35)
                c.line(cursor_x, y - height, cursor_x, y)
            cursor_x += col_width
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(x, y - height, x + sum(widths), y - height)
        y -= height
    return y


def _arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float,
           color: Color = GREEN, width: float = 1.7) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    if abs(y2 - y1) >= abs(x2 - x1):
        direction = 1 if y2 > y1 else -1
        c.line(x2, y2, x2 - 4, y2 - direction * 6)
        c.line(x2, y2, x2 + 4, y2 - direction * 6)
    else:
        direction = 1 if x2 > x1 else -1
        c.line(x2, y2, x2 - direction * 6, y2 - 4)
        c.line(x2, y2, x2 - direction * 6, y2 + 4)


def _callout(c: canvas.Canvas, x: float, y: float, w: float, h: float,
             title: str, body: str, accent: Color = BLUE, body_size: float = 9) -> None:
    _rounded(c, x, y, w, h, PANEL)
    c.setFillColor(accent)
    c.roundRect(x, y, 5, h, 2, fill=1, stroke=0)
    _text(c, x + 15, y + h - 21, title, 10, WHITE, "Helvetica-Bold")
    _paragraph(c, x + 15, y + h - 39, body, w - 29, body_size, body_size * 1.32, MUTED)


def _draw_cover(c: canvas.Canvas, root: Path) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    _text(c, MARGIN, PAGE_H - 72, "ROBOTICS PORTFOLIO | PROJECT GUIDE", 9, GREEN, "Helvetica-Bold")
    _text(c, MARGIN, PAGE_H - 122, "Camera-Based", 30, WHITE, "Helvetica-Bold")
    _text(c, MARGIN, PAGE_H - 160, "Visual Servoing", 30, WHITE, "Helvetica-Bold")
    _text(c, MARGIN, PAGE_H - 190, "A project explainer for visual-servo-mujoco", 12, MUTED)
    image_path = root / "assets" / "camera_view.png"
    image = ImageReader(str(image_path))
    image_x, image_y, image_w, image_h = MARGIN, 260, CONTENT_W, CONTENT_W * 0.75
    c.setFillColor(WHITE)
    c.roundRect(image_x - 3, image_y - 3, image_w + 6, image_h + 6, 10, fill=1, stroke=0)
    c.drawImage(image, image_x, image_y, width=image_w, height=image_h, preserveAspectRatio=True, mask="auto")
    _text(c, MARGIN, 221, "Pixels -> a table-plane estimate -> joint motion -> a measured result", 10.5, WHITE, "Helvetica-Bold")
    _paragraph(
        c,
        MARGIN,
        198,
        "This guide explains the important code, what the benchmarks actually establish, "
        "which experiments were rejected, and the next evidence-based milestone.",
        CONTENT_W,
        9.5,
        13,
        MUTED,
    )
    chips = [("MuJoCo", BLUE), ("OpenCV", GREEN), ("Image Jacobian", ORANGE), ("Paired benchmarks", BLUE)]
    x = MARGIN
    for label, color in chips:
        w = 91 if len(label) < 10 else 116
        _rounded(c, x, 94, w, 27, PANEL, stroke=GRID, radius=9)
        _text(c, x + 10, 103, label, 8, color, "Helvetica-Bold")
        x += w + 8
    _text(c, MARGIN, 48, "Local work on feat/independent-benchmark-review | Updated 20 Oct 2026", 8, MUTED)
    c.showPage()


def _draw_pipeline(c: canvas.Canvas) -> None:
    _header(c, 2, "System at a glance")
    y = _title(
        c,
        "The system in one picture",
        "A small simulation that makes every perception and control step inspectable.",
    )
    nodes = [
        ("01", "Render", "MuJoCo produces a top-down RGB image of the table, target, and arm.", BLUE),
        ("02", "Detect", "OpenCV segments the red target and green end-effector marker.", GREEN),
        ("03", "Estimate", "A pinhole camera model maps target pixels to table coordinates.", ORANGE),
        ("04", "Move", "Inverse kinematics or image feedback updates the two arm joints.", BLUE),
        ("05", "Score", "MuJoCo ground truth measures final distance after control stops.", GREEN),
    ]
    card_x, card_w, card_h = 73, CONTENT_W - 73, 88
    top = y - 4
    for index, (number, label, body, accent) in enumerate(nodes):
        card_y = top - (index + 1) * card_h - index * 10
        _rounded(c, card_x, card_y, card_w, card_h, PANEL)
        c.setFillColor(accent)
        c.circle(47, card_y + card_h / 2, 16, fill=1, stroke=0)
        c.setFillColor(BG)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(47, card_y + card_h / 2 - 3, number)
        _text(c, card_x + 17, card_y + card_h - 25, label, 13, WHITE, "Helvetica-Bold")
        _paragraph(c, card_x + 17, card_y + card_h - 44, body, card_w - 34, 9, 12, MUTED)
        if index < len(nodes) - 1:
            _arrow(c, 47, card_y - 1, 47, card_y - 10, GREEN)
    _callout(
        c,
        MARGIN,
        49,
        CONTENT_W,
        67,
        "What is measured and what is not",
        "The controller receives rendered pixels and joint state. Ground-truth target position is read after the run for scoring; it is not fed into the controller.",
        ORANGE,
        8.6,
    )
    c.showPage()


def _draw_control(c: canvas.Canvas) -> None:
    _header(c, 3, "Control loop")
    _title(
        c,
        "What one feedback update does",
        "Image feedback corrects pixel error with a local model of how each joint moves the image.",
    )
    x, y, w, h = MARGIN, 586, CONTENT_W, 116
    _rounded(c, x, y, w, h, PANEL_ALT)
    _section_label(c, x + 17, y + h - 23, "Damped image-space update", GREEN)
    _text(c, x + 18, y + 54, "e = target_pixel - tip_pixel", 16, WHITE, "Helvetica-Bold")
    _text(c, x + 18, y + 29, "dq = 0.5 J^T (J J^T + I)^-1 e", 15, BLUE, "Helvetica-Bold")
    _text(c, x + 18, y + 11, "then clip each joint change to 0.30 rad and advance 0.4 simulated seconds", 8.5, MUTED)

    steps = [
        ("Measure", "Find red target and green tip pixels."),
        ("Compare", "Compute the 2D image-space error."),
        ("Solve", "Use the Jacobian and damping term to get dq."),
        ("Advance", "Move, render again, and count a fresh check."),
    ]
    step_w = (CONTENT_W - 24) / 4
    y0 = 462
    for index, (label, body) in enumerate(steps):
        sx = MARGIN + index * (step_w + 8)
        _rounded(c, sx, y0, step_w, 91, PANEL)
        _text(c, sx + 11, y0 + 67, f"{index + 1}. {label}", 10, GREEN, "Helvetica-Bold")
        _paragraph(c, sx + 11, y0 + 49, body, step_w - 20, 8.3, 11, MUTED)
        if index < 3:
            _arrow(c, sx + step_w + 1, y0 + 45, sx + step_w + 7, y0 + 45, ORANGE, 1.2)

    _callout(
        c,
        MARGIN,
        329,
        CONTENT_W,
        103,
        "Why the damping term is there",
        "The arm has two joints but the observed error has two pixel coordinates. Near a singular pose, a raw inverse can demand a huge joint change. Adding I stabilizes the solve, and the per-joint cap limits one update.",
        BLUE,
    )
    _callout(
        c,
        MARGIN,
        199,
        CONTENT_W,
        103,
        "Two different tip sources in experiments",
        "The standard image-feedback controller detects the tip in the camera image. The timing benchmark can instead project the current joint angles through the calibrated geometry. That is current proprioceptive state paired with a delayed target image, not fully delayed vision.",
        ORANGE,
    )
    _text(c, MARGIN, 155, "Important constants in run.py", 10, WHITE, "Helvetica-Bold")
    _paragraph(
        c,
        MARGIN,
        137,
        "8 px tolerance | 3 consecutive checks | 20 maximum motion updates | 5 missing-target observations of grace | 4.5 cm physical success radius",
        CONTENT_W,
        9,
        12,
        MUTED,
    )
    _text(c, MARGIN, 87, "The controller stop reason and the physical reach score are separate fields in each trial.", 9, GREEN, "Helvetica-Bold")
    c.showPage()


def _draw_code_map(c: canvas.Canvas) -> None:
    _header(c, 4, "Code tour")
    _title(
        c,
        "Where the important code lives",
        "The core loop is intentionally small. Benchmarks isolate one assumption at a time.",
    )
    entries = [
        ("model.py", "MODEL_XML", "Inline scene: camera, table, two-link arm, target, and position actuators."),
        ("controller.py", "projection + IK + Jacobian", "Camera geometry, forward/inverse kinematics, pixel Jacobian, damped joint update, and opt-in predictor functions."),
        ("run.py", "run_trials", "Detection, seeded targets, control loop, dropouts, camera delay, visibility trace, stop reason, physical scoring, and demo recording."),
        ("benchmark.py", "paired robustness suite", "Same seeded targets across open loop and image feedback; aggregates success and error percentiles."),
        ("dynamic_benchmark.py", "moving target + latency", "Separates target motion, observation age, and strict vs experimental stopping."),
        ("prediction_benchmark.py / reversal_benchmark.py", "opt-in predictor tests", "Compares measured target with extrapolators while keeping current joint-state tip projection explicit."),
        ("sensor timing + timestamped verification", "feedback + completion", "Compares fresh, fully delayed, and hybrid control inputs; tests post-stop camera age and settling."),
        ("motion_envelope_benchmark.py / conditional_uncertainty_benchmark.py", "held-out completion", "Calibrates global or pixel-feature q99 margins on training seeds, then scores fresh visual decisions on held-out camera replays."),
        ("verification_*.py / terminal_verification.py", "fresh-view evidence + quality", "Renders a fresh terminal view, scores marker quality, and measures calibration sensitivity, coverage, and decision errors."),
        ("camera_calibration_*.py", "landmark health + transfer", "Matches fixed landmarks to pinhole projections, freezes a threshold, then evaluates pose transfer, visibility, and radial distortion."),
        ("public_camera_calibration.py / camera_distortion.py", "fit and correct radial lens error", "Fits k1,k2 on recorded checkerboard views; applies optional pixel correction and benchmarks its effect on reaching."),
        ("perception_candidate.py / perception_candidate_validation.py", "circle prior + held-out test", "Keeps the largest-contour baseline and compares an opt-in size-prior detector with abstention on randomized rendered scenes."),
    ]
    y = PAGE_H - 141
    for index, (file_name, function, explanation) in enumerate(entries):
        card_h = 46
        _rounded(c, MARGIN, y - card_h, CONTENT_W, card_h, PANEL if index % 2 == 0 else PANEL_ALT)
        _text(c, MARGIN + 13, y - 16, file_name, 8.7, BLUE, "Helvetica-Bold")
        _right_text(c, PAGE_W - MARGIN - 13, y - 17, function, 7.6, GREEN, "Helvetica-Bold")
        _paragraph(c, MARGIN + 13, y - 29, explanation, CONTENT_W - 26, 7.8, 9.3, MUTED)
        y -= card_h + 2
    _callout(
        c,
        MARGIN,
        48,
        CONTENT_W,
        61,
        "Good first files to read",
        "Start with model.py, then controller.py, then run.py around run_trials and the image_feedback branch. Read benchmark modules after you can trace one episode.",
        GREEN,
        8.4,
    )
    c.showPage()


def _draw_benchmarks(c: canvas.Canvas) -> None:
    _header(c, 5, "Measured failures")
    _title(
        c,
        "What the current controller handles",
        "Image feedback corrects several errors, but noisy sensing and visibility still expose real failure modes.",
    )
    _section_label(c, MARGIN, 637, "Physical success by target-pixel noise")
    chart_x, chart_y, chart_w, chart_h = MARGIN + 22, 472, CONTENT_W - 42, 145
    values = [("Open loop", [100, 100, 92, 34], BLUE), ("Image feedback", [100, 100, 100, 88], GREEN)]
    labels = ["0 px", "2 px", "8 px", "20 px"]
    group_w = chart_w / 4
    for group_index, label in enumerate(labels):
        gx = chart_x + group_index * group_w
        for series_index, (series, vals, color) in enumerate(values):
            bar_w = 21
            bx = gx + 24 + series_index * 26
            bh = chart_h * vals[group_index] / 100
            c.setFillColor(color)
            c.roundRect(bx, chart_y, bar_w, bh, 4, fill=1, stroke=0)
            _text(c, bx - 2, chart_y + bh + 5, f"{vals[group_index]}%", 7.2, WHITE)
        _text(c, gx + 23, chart_y - 15, label, 8, MUTED)
    _text(c, chart_x, chart_y - 34, "Blue: open loop   Green: image feedback   |   100 trials per condition", 8.2, MUTED)

    _callout(
        c,
        MARGIN,
        333,
        CONTENT_W,
        95,
        "At 20 px noise",
        "The default feedback controller reached 88/100 targets within 4.5 cm; open loop reached 34/100. Feedback median error was 2.12 cm and P95 was 6.39 cm. The controller uses a three-sample median, but this does not eliminate large-noise failures.",
        GREEN,
    )
    _section_label(c, MARGIN, 304, "Dropout length separates recovery from timeout", ORANGE)
    rows = [
        ["Injected blackout", "Physical success", "Reacquired after gap"],
        ["1 observation", "100/100", "100/100"],
        ["3 observations", "100/100", "100/100"],
        ["5 observations", "100/100", "100/100"],
        ["6 observations", "33/100", "0/100"],
    ]
    end_y = _table(c, MARGIN, 285, [160, 140, 170], rows, font_size=8.2)
    _paragraph(
        c,
        MARGIN,
        end_y - 18,
        "Ground-truth distance decides physical success. The controller can stop without fresh vision even when the arm is physically close.",
        CONTENT_W,
        8.8,
        12,
        MUTED,
    )
    _callout(
        c,
        MARGIN,
        52,
        CONTENT_W,
        62,
        "Natural self-occlusion is the next measured gap",
        "In the clean strict run, 91/100 trials reached physically but stopped after the arm hid the target from the overhead camera.",
        RED,
        8.4,
    )
    c.showPage()


def _draw_sensor_timing(c: canvas.Canvas) -> None:
    _header(c, 6, "Delayed vision")
    _title(
        c,
        "Three sensor setups that must stay distinct",
        "The timing experiments separate image age from the source used to locate the arm tip.",
    )
    boxes = [
        ("Fresh RGB", "target pixel now\ntip pixel now", BLUE),
        ("Fully delayed RGB", "target pixel delayed\ntip pixel delayed", ORANGE),
        ("Vision + proprioception", "target pixel delayed\ntip from current joints", GREEN),
    ]
    box_w = (CONTENT_W - 22) / 3
    for i, (name, detail, color) in enumerate(boxes):
        bx = MARGIN + i * (box_w + 11)
        _rounded(c, bx, 596, box_w, 106, PANEL_ALT)
        c.setFillColor(color)
        c.rect(bx, 596, box_w, 5, fill=1, stroke=0)
        _text(c, bx + 12, 675, name, 10.2, WHITE, "Helvetica-Bold")
        _paragraph(c, bx + 12, 650, detail.replace("\n", " | "), box_w - 24, 8.5, 11, MUTED)
    _callout(
        c,
        MARGIN,
        512,
        CONTENT_W,
        61,
        "How to read the held-out results",
        "Each row uses seeds 100-104 and 100 trials. The delta is paired median reaching error: hybrid minus fully delayed RGB. A negative value means lower error in this simulator.",
        GREEN,
        8.2,
    )
    rows = [
        ["Target path", "Noise", "Delay", "Delayed RGB", "Hybrid tip"],
        ["Sinusoidal", "0 px", "0.4 s", "4%", "84%"],
        ["Sinusoidal", "0 px", "0.8 s", "63%", "81%"],
        ["Sinusoidal", "2 px", "0.4 s", "11%", "68%"],
        ["Sinusoidal", "2 px", "0.8 s", "56%", "85%"],
        ["Piecewise linear", "0 px", "0.4 s", "34%", "88%"],
        ["Piecewise linear", "0 px", "0.8 s", "58%", "93%"],
        ["Piecewise linear", "2 px", "0.4 s", "37%", "83%"],
        ["Piecewise linear", "2 px", "0.8 s", "56%", "95%"],
    ]
    bottom = _table(c, MARGIN, 495, [146, 66, 66, 112, 112], rows, font_size=7.7, leading=9.2)
    _callout(
        c,
        MARGIN,
        121,
        CONTENT_W,
        96,
        "What improved, and what did not",
        "The current-joint-state tip hybrid improved physical success in all eight held-out settings. Its P95 error was 4.92-6.71 cm. It still had 0/100 current-time visual confirmations in every delayed condition. Physical reaching and fresh visual completion remain separate outcomes.",
        ORANGE,
        8.6,
    )
    _paragraph(
        c,
        MARGIN,
        97,
        "The paired bootstrap resamples only five seed groups. The interval reflects these simulator runs, not broad generalization.",
        CONTENT_W,
        8,
        10,
        MUTED,
    )
    c.showPage()


def _draw_timestamped_verification(c: canvas.Canvas, root: Path) -> None:
    report_path = root / "artifacts" / "timestamped-verification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result_rows = report["results"]
    trial_rows = report["trial_results"]
    policies = (
        ("arrival_first_visible_frame", None, "Arrival only", RED),
        ("two_post_settle_frames", 0.5, "Two frames <= 0.5 s", BLUE),
        ("two_post_settle_frames", 1.0, "Two frames <= 1.0 s", GREEN),
    )
    delivery_profiles = (
        ("near_real_time", "~0.05 s"),
        ("moderate_latency", "~0.40 s"),
        ("high_latency", "~0.80 s"),
    )

    def metrics(delivery: str, policy: str, age: float | None) -> tuple[float, float]:
        selected = [
            row
            for row in result_rows
            if row["delivery_profile"] == delivery
            and row["policy"] == policy
            and row["max_frame_age_s"] == age
        ]
        trials = sum(row["trials"] for row in selected)
        if not trials:
            return 0.0, 0.0
        confirmations = sum(row["visually_confirmed"] for row in selected)
        false_at_delivery = sum(
            row["false_confirmations_at_delivery"] for row in selected
        )
        return 100 * confirmations / trials, 100 * false_at_delivery / trials

    _header(c, 7, "Timestamped completion")
    _title(
        c,
        "A stopped arm needs time-correct visual evidence",
        "The controller's reaching result and a camera's later completion report are separate events. This benchmark replays the stopped arm while a delayed camera continues observing a moving target.",
    )

    timeline = [
        ("Controller stops", "Save measured joint pose and velocity; hold the pose."),
        ("Arm settles", "Joint speed <0.02 rad/s for 0.1 s; within 0.8 s."),
        ("Camera captures", "5 Hz stream; require two consecutive post-settle frames."),
        ("Receiver decides", "Check frame age and deliver by a 2 s deadline."),
    ]
    card_y = 604
    card_h = 82
    gap = 9
    card_w = (CONTENT_W - 3 * gap) / 4
    for index, (label, detail) in enumerate(timeline):
        x = MARGIN + index * (card_w + gap)
        _rounded(c, x, card_y, card_w, card_h, PANEL_ALT)
        _text(c, x + 9, card_y + 59, label, 8.2, WHITE, "Helvetica-Bold")
        _paragraph(c, x + 9, card_y + 43, detail, card_w - 18, 7.0, 9.0, MUTED)
        if index < len(timeline) - 1:
            _arrow(c, x + card_w + 1, card_y + card_h / 2, x + card_w + gap - 1, card_y + card_h / 2, GREEN, 1.2)

    _section_label(c, MARGIN, 574, "Outcome rates across all motion profiles and control modes", ORANGE)

    def draw_chart(
        *,
        chart_x: float,
        title: str,
        value_index: int,
        scale_max: float,
        tick_format: str,
    ) -> None:
        chart_w = (CONTENT_W - 26) / 2
        plot_left = chart_x + 31
        plot_right = chart_x + chart_w - 5
        plot_top = 535
        plot_bottom = 373
        plot_h = plot_top - plot_bottom
        _text(c, chart_x, 553, title, 9.0, WHITE, "Helvetica-Bold")
        c.setStrokeColor(GRID)
        c.setLineWidth(0.45)
        for tick in range(6):
            value = scale_max * tick / 5
            y = plot_bottom + plot_h * tick / 5
            c.line(plot_left, y, plot_right, y)
            label = f"{value:.0f}%" if tick_format == "percent" else f"{value:.1f}"
            _right_text(c, plot_left - 5, y - 2, label, 6.4, MUTED)

        group_w = (plot_right - plot_left) / len(delivery_profiles)
        bar_w = 12
        bar_gap = 3
        cluster_w = len(policies) * bar_w + (len(policies) - 1) * bar_gap
        for group_index, (delivery, label) in enumerate(delivery_profiles):
            center = plot_left + group_w * (group_index + 0.5)
            _text(c, center - 22, plot_bottom - 16, label, 6.8, MUTED)
            for policy_index, (policy, age, _, color) in enumerate(policies):
                percent, false_rate = metrics(delivery, policy, age)
                value = (percent, false_rate)[value_index]
                height = plot_h * min(value, scale_max) / scale_max
                x = center - cluster_w / 2 + policy_index * (bar_w + bar_gap)
                c.setFillColor(color)
                c.roundRect(x, plot_bottom, bar_w, max(1.0, height), 2, fill=1, stroke=0)

    max_false_rate = max(
        (
            metrics(delivery, policy, age)[1]
            for delivery, _ in delivery_profiles
            for policy, age, _, _ in policies
        ),
        default=0.0,
    )
    false_scale = max(1.0, float((int(max_false_rate / 5) + 1) * 5))
    draw_chart(
        chart_x=MARGIN,
        title="Visual confirmation at delivery",
        value_index=0,
        scale_max=100.0,
        tick_format="percent",
    )
    draw_chart(
        chart_x=MARGIN + (CONTENT_W + 26) / 2,
        title="Outside 4.5 cm at delivery per 100 trials",
        value_index=1,
        scale_max=false_scale,
        tick_format="count",
    )

    legend_y = 337
    for index, (_, _, label, color) in enumerate(policies):
        x = MARGIN + index * 169
        c.setFillColor(color)
        c.roundRect(x, legend_y - 1, 9, 9, 2, fill=1, stroke=0)
        _text(c, x + 15, legend_y, label, 7.4, MUTED)
    protocol = report["protocol"]
    _text(
        c,
        MARGIN,
        309,
        (
            f"{protocol['control_trial_count']:,} controller trials | "
            f"{protocol['policy_trial_count']:,} policy outcomes | seeds "
            f"{protocol['base_seed']}-{protocol['base_seed'] + protocol['seeds'] - 1} | "
            "four start phases per motion path"
        ),
        7.2,
        MUTED,
    )

    age_half = [
        row
        for row in result_rows
        if row["policy"] == "two_post_settle_frames"
        and row["max_frame_age_s"] == 0.5
    ]
    age_one = [
        row
        for row in result_rows
        if row["policy"] == "two_post_settle_frames"
        and row["max_frame_age_s"] == 1.0
    ]
    half_false = sum(row["false_confirmations_at_delivery"] for row in age_half)
    one_false = sum(row["false_confirmations_at_delivery"] for row in age_one)
    half_capture_close = sum(
        row["false_confirmation_at_delivery"]
        and row["physical_error_at_decision_capture_m"]
        <= report["protocol"]["physical_success_threshold_m"]
        for row in trial_rows
        if row["policy"] == "two_post_settle_frames"
        and row["max_frame_age_s"] == 0.5
    )
    one_capture_close = sum(
        row["false_confirmation_at_delivery"]
        and row["physical_error_at_decision_capture_m"]
        <= report["protocol"]["physical_success_threshold_m"]
        for row in trial_rows
        if row["policy"] == "two_post_settle_frames"
        and row["max_frame_age_s"] == 1.0
    )
    _callout(
        c,
        MARGIN,
        209,
        CONTENT_W,
        82,
        "The age limit changes coverage and delivery-time reliability",
        (
            f"Across {sum(row['trials'] for row in age_half):,} 0.5 s guarded outcomes, "
            f"{half_false} image-positive decisions arrived after the target-arm distance "
            f"left the success radius; {half_capture_close} were physically within the "
            f"radius at capture. At 1.0 s, {one_false} mismatches occurred, and "
            f"{one_capture_close} were within the radius at capture. Most mismatches "
            "therefore reflect target motion during transport, not an incorrect image at "
            "capture. These age limits remain measured tradeoffs, not validated defaults."
        ),
        ORANGE,
        8.0,
    )
    _callout(
        c,
        MARGIN,
        64,
        CONTENT_W,
        119,
        "Keep the two outcomes and the three clocks explicit",
        "Physical reaching is scored at controller stop from MuJoCo state. Visual confirmation uses camera estimates from frames captured after the arm settles. Delivery-time distance is an evaluation label only; it is never sent to the detector, controller, or confirmation rule. This is an offline benchmark and does not change the default controller. The next investigation should test conservative task-completion semantics across target speeds, frame ages, and transport loss before any runtime policy is promoted.",
        BLUE,
        8.2,
    )
    c.showPage()


def _draw_completion_speed(c: canvas.Canvas, root: Path) -> None:
    report = json.loads(
        (root / "artifacts" / "timestamped-completion-speed.json").read_text(
            encoding="utf-8"
        )
    )
    result_rows = report["results"]
    trial_rows = report["trial_results"]
    conditions = report["protocol"]["motion_conditions"]
    deliveries = (
        ("near_real_time", "0.05 s"),
        ("moderate_latency", "0.40 s"),
        ("high_latency", "0.80 s"),
    )

    def rate(condition: str, delivery: str, age: float, field: str) -> tuple[float, int, int]:
        selected = [
            row
            for row in result_rows
            if row["motion_condition"] == condition
            and row["delivery_profile"] == delivery
            and row["policy"] == "two_post_settle_frames"
            and row["max_frame_age_s"] == age
        ]
        trials = sum(row["trials"] for row in selected)
        count = sum(row[field] for row in selected)
        return (100 * count / trials if trials else 0.0, count, trials)

    names = {"stationary": "Stationary", "sinusoidal_0p25_hz": "Sine 0.25 Hz", "sinusoidal_0p50_hz": "Sine 0.50 Hz", "piecewise_linear_0p50_hz": "Piecewise 0.50 Hz"}
    _header(c, 8, "Completion across target speeds")
    _title(
        c,
        "Target motion makes stale camera decisions visible",
        "The follow-on benchmark holds the stop, settling, camera, and confirmation pipeline fixed while varying target path and analytic peak speed.",
    )
    mismatches_half = sum(
        row["false_confirmations_at_delivery"]
        for row in result_rows
        if row["policy"] == "two_post_settle_frames" and row["max_frame_age_s"] == 0.5
    )
    capture_close_half = sum(
        row["physical_success_at_decision_capture"] is True
        and row["false_confirmation_at_delivery"]
        for row in trial_rows
        if row["policy"] == "two_post_settle_frames" and row["max_frame_age_s"] == 0.5
    )
    mismatches_one = sum(
        row["false_confirmations_at_delivery"]
        for row in result_rows
        if row["policy"] == "two_post_settle_frames" and row["max_frame_age_s"] == 1.0
    )
    capture_close_one = sum(
        row["physical_success_at_decision_capture"] is True
        and row["false_confirmation_at_delivery"]
        for row in trial_rows
        if row["policy"] == "two_post_settle_frames" and row["max_frame_age_s"] == 1.0
    )
    _callout(
        c,
        MARGIN,
        660,
        CONTENT_W,
        63,
        "Most stale positives began during transport",
        f"For the 0.5 s age guard, {capture_close_half}/{mismatches_half} positive decisions were physically inside the radius at image capture. For the 1.0 s guard, {capture_close_one}/{mismatches_one} were close at capture. Every stationary-target guard cell had zero delivery mismatches.",
        ORANGE,
        7.6,
    )

    _section_label(c, MARGIN, 632, "Two-frame policy: confirmation and stale-at-delivery rates", BLUE)
    label_w = 92
    gap = 12
    panel_w = (CONTENT_W - label_w - gap) / 2
    left_x = MARGIN + label_w
    right_x = left_x + panel_w + gap
    cell_w = panel_w / 3
    _text(c, left_x + panel_w / 2, 609, "Maximum frame age 0.5 s", 8.7, WHITE, "Helvetica-Bold")
    _text(c, right_x + panel_w / 2, 609, "Maximum frame age 1.0 s", 8.7, WHITE, "Helvetica-Bold")
    for panel_x in (left_x, right_x):
        for index, (_, delivery_label) in enumerate(deliveries):
            _text(c, panel_x + cell_w * (index + 0.5) - 14, 592, delivery_label, 7, MUTED)

    y_top = 580
    row_h = 42
    for row_index, condition in enumerate(conditions):
        y = y_top - (row_index + 1) * row_h
        speed = condition["analytic_peak_speed_m_s"]
        _text(c, MARGIN, y + 20, names[condition["name"]], 6.6, WHITE, "Helvetica-Bold")
        _text(c, MARGIN, y + 9, f"{speed:.3f} m/s", 6.2, MUTED)
        for panel_index, age in enumerate((0.5, 1.0)):
            panel_x = left_x if panel_index == 0 else right_x
            for delivery_index, (delivery, _) in enumerate(deliveries):
                confirm, confirmed, trials = rate(
                    condition["name"], delivery, age, "visually_confirmed"
                )
                stale, false_count, _ = rate(
                    condition["name"], delivery, age, "false_confirmations_at_delivery"
                )
                x = panel_x + delivery_index * cell_w
                fill = Color(0.08, 0.15 + 0.28 * confirm / 100, 0.20 + 0.22 * confirm / 100)
                c.setFillColor(fill)
                c.setStrokeColor(GRID)
                c.roundRect(x + 1, y + 2, cell_w - 3, row_h - 4, 4, fill=1, stroke=1)
                _text(c, x + 7, y + 23, f"confirm {confirm:.1f}%", 6.2, WHITE, "Helvetica-Bold")
                _text(c, x + 7, y + 11, f"out {stale:.1f}% ({false_count}/{trials})", 5.8, ORANGE)

    _callout(
        c,
        MARGIN,
        315,
        CONTENT_W,
        75,
        "A fixed age cap is not a general completion rule",
        "The 0.5 s cap confirmed no high-latency outcomes, while 1.0 s restored coverage and allowed motion-related mismatches. Reliability was not monotonic in peak speed: the 0.25 Hz sinusoid had more stale delivery positives than either fast path. Path shape and starting phase also matter.",
        RED,
        7.6,
    )
    reached = []
    for condition in conditions:
        selected = [
            row
            for row in trial_rows
            if row["motion_condition"] == condition["name"]
            and row["policy"] == "arrival_first_visible_frame"
            and row["delivery_profile"] == "near_real_time"
        ]
        reached.append(
            f"{names[condition['name']]} {sum(row['physical_success_at_controller_stop'] for row in selected)}/{len(selected)}"
        )
    _callout(
        c,
        MARGIN,
        212,
        CONTENT_W,
        78,
        "Keep reaching separate from visual completion",
        f"Controller-stop physical reaches: {'; '.join(reached)}. The rate heatmap instead counts fresh camera decisions. The five controller modes remain distinct in CSV; only this chart pools them within each condition and delivery profile. Delivery-time ground truth is evaluation-only.",
        BLUE,
        7.5,
    )
    _paragraph(
        c,
        MARGIN,
        185,
        "The 975 controller trials use seeds 950–954 and produce 8,775 policy outcomes. Camera motion, pixel noise, frame loss, settling, and the controller are unchanged from the previous timestamped benchmark. Results cover one target amplitude and five seed groups; no real camera or hardware was tested.",
        CONTENT_W,
        7.4,
        9.5,
        MUTED,
    )
    c.showPage()


def _draw_negative_results(c: canvas.Canvas) -> None:
    _header(c, 9, "Experimental predictors")
    _title(
        c,
        "A useful negative result: prediction lost to measurement",
        "Moving-target latency made extrapolation tempting. Paired evidence says these predictors stay experimental.",
    )
    _section_label(c, MARGIN, 638, "Sinusoidal target | 0.4 s delay | noise-free", ORANGE)
    chart_x, chart_y, chart_w, chart_h = MARGIN + 22, 464, CONTENT_W - 44, 146
    series = [("Measured", 85, GREEN), ("Constant velocity", 10, BLUE), ("Reversal aware", 33, ORANGE)]
    bw, gap = 70, 64
    total = 3 * bw + 2 * gap
    start = chart_x + (chart_w - total) / 2
    for i, (name, value, color) in enumerate(series):
        x = start + i * (bw + gap)
        h = chart_h * value / 100
        c.setFillColor(color)
        c.roundRect(x, chart_y, bw, h, 6, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(x + bw / 2, chart_y + h + 8, f"{value}%")
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8)
        c.drawCentredString(x + bw / 2, chart_y - 15, name)
    _text(c, chart_x, chart_y - 34, "Success within 4.5 cm | five seeds x 20 targets", 8, MUTED)

    rows = [
        ["Method", "Median reaching error", "Success"],
        ["Measured target", "2.70 cm", "85%"],
        ["Constant velocity", "6.48 cm", "10%"],
        ["Reversal aware", "6.44 cm", "33%"],
    ]
    bottom = _table(c, MARGIN, 411, [190, 210, 100], rows, font_size=8.1)
    _callout(
        c,
        MARGIN,
        205,
        CONTENT_W,
        95,
        "Across all eight reversal benchmark settings",
        "The reversal-aware predictor increased median error in every setting; each seed-cluster bootstrap interval for the paired error delta was above zero. It often beat plain velocity prediction, but never beat the measured-target baseline on success. Keep predictors opt-in and preserve the baseline.",
        RED,
    )
    _paragraph(
        c,
        MARGIN,
        185,
        "At 0.8 s delay the reversal-aware method reached only 5% on sinusoidal noise-free trials versus 85% for measured target. The measured result rules out this simple reversal guard as the default latency fix.",
        CONTENT_W,
        9,
        12.5,
        MUTED,
    )
    _text(c, MARGIN, 145, "Other measured tradeoffs", 10, WHITE, "Helvetica-Bold")
    _paragraph(
        c,
        MARGIN,
        126,
        "The three-sample median helped 20 px noise versus raw feedback, while larger windows raised clean error. Stale-target stopping did not improve physical success and can stop without reacquisition. These remain scoped experiments, not new defaults.",
        CONTENT_W,
        9,
        12.5,
        MUTED,
    )
    c.showPage()


def _draw_verifier(c: canvas.Canvas) -> None:
    _header(c, 10, "Second-view benchmark")
    _title(
        c,
        "A second view can supply a separate visual estimate",
        "Four oblique candidates were evaluated; an opt-in east-view check now runs after the overhead controller stops.",
    )
    # Simple top-down and side-view camera geometry sketch.
    x0, y0 = MARGIN + 10, 580
    _rounded(c, x0, y0, 226, 122, PANEL)
    _section_label(c, x0 + 13, y0 + 102, "Overhead control view", BLUE)
    c.setStrokeColor(HexColor("#B6A27B"))
    c.setLineWidth(2)
    c.line(x0 + 38, y0 + 42, x0 + 153, y0 + 42)
    c.setFillColor(ORANGE)
    c.circle(x0 + 165, y0 + 43, 7, fill=1, stroke=0)
    c.setFillColor(GREEN)
    c.circle(x0 + 167, y0 + 46, 6, fill=1, stroke=0)
    c.setFillColor(SUBTLE)
    c.circle(x0 + 99, y0 + 43, 17, fill=1, stroke=0)
    c.setStrokeColor(RED)
    c.line(x0 + 165, y0 + 53, x0 + 165, y0 + 89)
    _text(c, x0 + 12, y0 + 17, "The arm can cover the target from above.", 8, MUTED)

    x1 = x0 + 244
    _rounded(c, x1, y0, 226, 122, PANEL)
    _section_label(c, x1 + 13, y0 + 102, "East oblique candidate", GREEN)
    c.setStrokeColor(HexColor("#B6A27B"))
    c.setLineWidth(2)
    c.line(x1 + 35, y0 + 37, x1 + 190, y0 + 37)
    c.setFillColor(GREEN)
    c.circle(x1 + 145, y0 + 67, 8, fill=1, stroke=0)
    c.setFillColor(ORANGE)
    c.circle(x1 + 163, y0 + 54, 8, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.circle(x1 + 30, y0 + 88, 7, fill=1, stroke=0)
    c.setStrokeColor(BLUE)
    c.setLineWidth(1)
    c.line(x1 + 34, y0 + 86, x1 + 140, y0 + 66)
    c.line(x1 + 34, y0 + 86, x1 + 160, y0 + 55)
    _text(c, x1 + 12, y0 + 17, "Pixels are projected to known marker heights.", 8, MUTED)

    rows = [
        ["View", "Control noise", "Both seen", "False success", "Missed success"],
        ["East", "0 px", "200/200", "0", "1"],
        ["East", "20 px", "198/200", "0", "14"],
        ["North", "0 px", "200/200", "0", "0"],
        ["North", "20 px", "200/200", "6", "11"],
    ]
    bottom = _table(c, MARGIN, 560, [90, 103, 103, 103, 103], rows, font_size=7.8)
    _callout(
        c,
        MARGIN,
        319,
        CONTENT_W,
        102,
        "Why east is the current candidate",
        "Across the two seed groups it produced no false success estimates and saw both markers in 198/200 noisy trials. It missed 14 true noisy-condition successes. The north view saw every marker but made six false success estimates. The conservative tradeoff is promising, not sufficient for a production stop rule.",
        GREEN,
        8.5,
    )
    _callout(
        c,
        MARGIN,
        194,
        CONTENT_W,
        91,
        "Scope of this evidence",
        "The oblique camera is rendered from the simulator. The controller still uses the overhead camera, and terminal verification remains a separate fresh observation after control stops. The next page measures estimator calibration offsets and controlled image changes.",
        ORANGE,
        8.5,
    )
    _paragraph(
        c,
        MARGIN,
        139,
        "The opt-in --verify-terminal path now renders the east camera after control stops. It reports the fresh image estimate separately; it never changes the controller stop or MuJoCo physical score. Missing markers make the estimate unavailable.",
        CONTENT_W,
        8.5,
        11,
        WHITE,
    )
    c.showPage()


def _draw_verifier_robustness(c: canvas.Canvas) -> None:
    _header(c, 11, "Verifier stress evidence")
    _title(
        c,
        "Calibration error changes the visual decision",
        "100 held-out paired trials per condition; each verifier stress leaves the reaching controller unchanged.",
    )

    card_y = 672
    card_w = (CONTENT_W - 16) / 3
    cards = [
        ("100", "paired trials per condition", BLUE),
        ("0 / 8", "controller outcome changes", GREEN),
        ("85 / 100", "physical reaches in every row", ORANGE),
    ]
    for index, (value, label, color) in enumerate(cards):
        x = MARGIN + index * (card_w + 8)
        _rounded(c, x, card_y - 55, card_w, 55, PANEL)
        _text(c, x + 12, card_y - 24, value, 16, color, "Helvetica-Bold")
        _text(c, x + 12, card_y - 42, label, 8, MUTED)

    rows = [
        ["Verifier condition", "Both seen / estimate", "False +", "Missed", "P95 error"],
        ["Nominal", "99 / 100", "0", "7", "1.74 cm"],
        ["Estimated FOV +5°", "99 / 100", "0", "15", "2.12 cm"],
        ["Estimated FOV -5°", "99 / 100", "1", "3", "1.66 cm"],
        ["Camera world X +2 cm", "99 / 100", "0", "7", "1.74 cm"],
        ["Estimated yaw +3°", "99 / 100", "0", "5", "1.84 cm"],
        ["Exposure ×0.25", "0 / 100", "n/a", "n/a", "n/a"],
        ["Exposure ×1.5", "99 / 100", "0", "10", "2.52 cm"],
        ["Target occluded", "0 / 100", "n/a", "n/a", "n/a"],
        ["Tip occluded", "0 / 100", "n/a", "n/a", "n/a"],
    ]
    table_bottom = _table(
        c,
        MARGIN,
        603,
        [145, 132, 65, 65, CONTENT_W - 407],
        rows,
        font_size=7.7,
    )

    _section_label(c, MARGIN, table_bottom - 25, "Missed physical successes among available estimates", ORANGE)
    bars = [
        ("Nominal", 8.3, "7 / 84"),
        ("FOV +5°", 17.9, "15 / 84"),
        ("FOV -5°", 3.6, "3 / 84"),
        ("Exposure ×1.5", 11.9, "10 / 84"),
    ]
    chart_y = table_bottom - 52
    bar_x = MARGIN + 112
    bar_max_w = 245
    for label, percentage, count in bars:
        _text(c, MARGIN, chart_y + 3, label, 8.2, WHITE)
        c.setFillColor(PANEL_ALT)
        c.roundRect(bar_x, chart_y, bar_max_w, 10, 4, fill=1, stroke=0)
        c.setFillColor(RED if percentage >= 15 else BLUE)
        c.roundRect(bar_x, chart_y, bar_max_w * percentage / 20, 10, 4, fill=1, stroke=0)
        _text(c, bar_x + bar_max_w + 9, chart_y + 2, f"{percentage:.1f}%  ({count})", 7.8, MUTED)
        chart_y -= 22

    _callout(
        c,
        MARGIN,
        49,
        CONTENT_W,
        91,
        "What these stresses do and do not establish",
        "A +5° FOV estimate doubled missed successes (7 to 15); -5° caused one false success. Dim exposure and marker occlusion yielded no estimate, so those trials have no decision score. Brightness is a uniform post-render multiplier and occlusion uses marker-color removal plus inpainting. This is a simulator stress test, not real-camera evidence.",
        ORANGE,
        8.1,
    )
    c.showPage()


def _draw_verifier_quality(c: canvas.Canvas) -> None:
    _header(c, 12, "Selective verifier")
    _title(
        c,
        "The quality gate trades coverage for fewer errors",
        "A heuristic marker score was swept on the same held-out trials; these results do not select a deployment threshold.",
    )

    feature_w = (CONTENT_W - 24) / 4
    features = ["Contour area", "Circularity", "Saturation", "Brightness"]
    for index, label in enumerate(features):
        x = MARGIN + index * (feature_w + 8)
        _rounded(c, x, 654, feature_w, 47, PANEL_ALT)
        _text(c, x + 10, 681, f"0{index + 1}", 8, GREEN, "Helvetica-Bold")
        _text(c, x + 10, 664, label, 8.5, WHITE, "Helvetica-Bold")
    _text(
        c,
        MARGIN,
        634,
        "Score = minimum component across both detected markers; it is a heuristic, not a probability.",
        8.5,
        MUTED,
    )

    rows = [
        ["Minimum score", "Coverage", "False +", "Missed", "Condition"],
        ["0.00", "99 / 100", "1", "3", "FOV estimate -5°"],
        ["0.15", "93 / 100", "1", "3", "FOV estimate -5°"],
        ["0.20", "39 / 100", "1", "3", "FOV estimate -5°"],
        ["0.50", "9 / 100", "0", "1", "FOV estimate -5°"],
    ]
    table_bottom = _table(
        c,
        MARGIN,
        602,
        [100, 100, 76, 76, CONTENT_W - 352],
        rows,
        font_size=8,
    )

    _section_label(c, MARGIN, table_bottom - 25, "Available estimates after applying the gate", BLUE)
    coverage = [("Gate 0.00", 99), ("Gate 0.15", 93), ("Gate 0.20", 39), ("Gate 0.50", 9)]
    bar_y = table_bottom - 53
    bar_x = MARGIN + 105
    bar_max_w = 310
    for index, (label, value) in enumerate(coverage):
        _text(c, MARGIN, bar_y + 3, label, 8.2, WHITE)
        c.setFillColor(PANEL_ALT)
        c.roundRect(bar_x, bar_y, bar_max_w, 11, 4, fill=1, stroke=0)
        c.setFillColor(RED if index == 3 else BLUE)
        c.roundRect(bar_x, bar_y, bar_max_w * value / 100, 11, 4, fill=1, stroke=0)
        _text(c, bar_x + bar_max_w + 9, bar_y + 2, f"{value}%", 8, MUTED)
        bar_y -= 26

    _callout(
        c,
        MARGIN,
        66,
        CONTENT_W,
        99,
        "A useful abstention point has not been found",
        "The 0.50 gate removed the single false success under a -5° FOV estimate, while leaving 9% coverage. At 0.20, coverage was 39% and the false success remained. Scores do not reveal calibration bias from marker appearance; no threshold is selected.",
        ORANGE,
        8.3,
    )
    c.showPage()


def _draw_frozen_quality_validation(c: canvas.Canvas) -> None:
    _header(c, 13, "Frozen threshold validation")
    _title(
        c,
        "The frozen quality gate failed its coverage target",
        "Threshold chosen on seeds 200-204, then evaluated unchanged on seeds 300-304; each split has 100 target trials paired across nine image conditions.",
    )

    _rounded(c, MARGIN, 643, CONTENT_W, 61, PANEL)
    _text(c, MARGIN + 15, 680, "Calibration rule", 9, GREEN, "Helvetica-Bold")
    _paragraph(
        c,
        MARGIN + 15,
        663,
        "Choose maximum candidate coverage with zero observed false-success decisions; require at least 50% coverage for a usable gate.",
        CONTENT_W - 30,
        8.5,
        11.5,
        MUTED,
    )

    _section_label(c, MARGIN, 612, "Independent validation results", BLUE)
    table_bottom = _table(
        c,
        MARGIN,
        594,
        [145, 110, 95, 72, CONTENT_W - 422],
        [
            ["Gate", "Accepted / candidate", "Coverage", "False +", "Missed"],
            ["Ungated 0.00", "600 / 600", "100%", "10", "48"],
            ["Frozen 0.60", "23 / 600", "3.8%", "0", "3"],
        ],
        font_size=8,
    )

    _section_label(c, MARGIN, table_bottom - 24, "Candidate-estimate coverage", BLUE)
    bar_x = MARGIN + 145
    bar_w = 280
    bars = [("Ungated", 100.0, BLUE), ("Frozen 0.60", 3.8, RED)]
    bar_y = table_bottom - 55
    for label, value, color in bars:
        _text(c, MARGIN, bar_y + 2, label, 8.4, WHITE)
        _rounded(c, bar_x, bar_y - 1, bar_w, 13, PANEL_ALT, radius=6)
        if value > 0:
            c.setFillColor(color)
            c.roundRect(bar_x, bar_y - 1, bar_w * value / 100, 13, 6, fill=1, stroke=0)
        _text(c, bar_x + bar_w + 10, bar_y + 2, f"{value:g}%", 8.2, MUTED)
        bar_y -= 31

    _callout(
        c,
        MARGIN,
        307,
        CONTENT_W,
        92,
        "The gate acted as a brightness filter",
        "All 23 accepted estimates came from bright exposure. It accepted none from nominal images or any camera calibration shift. Across all 900 validation condition-trials, 23 estimates were available (2.6%); dim and occluded images had no candidate detection before gating.",
        ORANGE,
        8.1,
    )

    _text(c, MARGIN, 275, "Paired held-out video", 9.3, WHITE, "Helvetica-Bold")
    _paragraph(
        c,
        MARGIN,
        256,
        "Seed 302, episode 17: fresh-view estimate 4.494 cm, MuJoCo physical error 4.6 cm. The 0.60 gate abstains; both clips preserve the same reaching outcome.",
        CONTENT_W,
        8.5,
        12,
        MUTED,
    )

    _callout(
        c,
        MARGIN,
        79,
        CONTENT_W,
        66,
        "Negative result: no usable threshold found",
        "The gate removed 10 false visual success decisions on validation but retained only 3.8% of candidate estimates, below the 50% usability floor. Zero observed errors do not establish a risk guarantee; no threshold is recommended.",
        RED,
        8,
    )
    c.showPage()


def _draw_camera_calibration_health(c: canvas.Canvas, root: Path) -> None:
    _header(c, 14, "Fixed-landmark calibration health")
    _title(
        c,
        "Independent landmarks expose camera-model drift",
        "A separate simulator diagnostic compares known tabletop points with image detections while the rendered camera moves away from nominal calibration.",
    )

    preview = root / "assets" / "demos" / "camera_calibration_health.png"
    image = ImageReader(str(preview))
    image_height = CONTENT_W * 544 / 1312
    c.drawImage(
        image,
        MARGIN,
        494,
        width=CONTENT_W,
        height=image_height,
        preserveAspectRatio=True,
        mask="auto",
    )
    _text(
        c,
        MARGIN,
        480,
        "White crosses: nominal projection   |   Green rings: detected centers   |   Orange lines: residuals",
        7.4,
        MUTED,
    )

    _section_label(c, MARGIN, 457, "Measured reprojection RMS (pixels; alarm line at 3 px)", BLUE)
    conditions = [
        ("Nominal", 0.51, GREEN),
        ("FOV +5 deg", 20.23, ORANGE),
        ("FOV -5 deg", 24.20, ORANGE),
        ("Camera X +2 cm", 6.84, ORANGE),
        ("Camera yaw +3 deg", 21.43, ORANGE),
        ("Combined camera shift", 32.02, ORANGE),
        ("Bright exposure x1.5", 0.45, GREEN),
        ("Dim exposure x0.25", None, RED),
    ]
    chart_x = MARGIN + 152
    chart_width = 285
    chart_top = 440
    chart_scale_max = 35.0
    threshold_x = chart_x + chart_width * 3 / chart_scale_max
    c.saveState()
    c.setStrokeColor(RED)
    c.setDash(2, 2)
    c.setLineWidth(0.8)
    c.line(threshold_x, 296, threshold_x, chart_top + 8)
    c.restoreState()
    for index, (label, value, color) in enumerate(conditions):
        y = chart_top - index * 18
        _text(c, MARGIN, y, label, 7.6, WHITE)
        c.setFillColor(PANEL_ALT)
        c.roundRect(chart_x, y - 2, chart_width, 9, 4, fill=1, stroke=0)
        if value is not None:
            c.setFillColor(color)
            c.roundRect(
                chart_x,
                y - 2,
                chart_width * value / chart_scale_max,
                9,
                4,
                fill=1,
                stroke=0,
            )
            _text(
                c,
                chart_x + chart_width * value / chart_scale_max + 6,
                y - 1,
                f"{value:.2f}",
                7.4,
                MUTED,
            )
        else:
            _text(c, chart_x + 6, y - 1, "unavailable: 0 / 4 landmarks", 7.4, RED)
    _text(c, threshold_x + 3, chart_top + 7, "3 px", 6.8, RED)

    flow_y = 224
    flow_labels = [
        ("Known table\ncoordinates", BLUE),
        ("Nominal pinhole\nprojection", GREEN),
        ("Magenta center\ndetections", ORANGE),
        ("Match + pixel\nRMS residual", BLUE),
    ]
    box_width = (CONTENT_W - 54) / 4
    for index, (label, color) in enumerate(flow_labels):
        x = MARGIN + index * (box_width + 18)
        _rounded(c, x, flow_y, box_width, 44, PANEL)
        _text(c, x + 9, flow_y + 27, f"0{index + 1}", 7.2, color, "Helvetica-Bold")
        lines = label.split("\n")
        _text(c, x + 30, flow_y + 26, lines[0], 7.1, WHITE, "Helvetica-Bold")
        _text(c, x + 30, flow_y + 13, lines[1], 7.1, WHITE, "Helvetica-Bold")
        if index < len(flow_labels) - 1:
            _arrow(c, x + box_width + 3, flow_y + 22, x + box_width + 13, flow_y + 22, MUTED, 1)

    _callout(
        c,
        MARGIN,
        67,
        CONTENT_W,
        91,
        "Promising detectability; threshold still exploratory",
        "Five renderer-side camera shifts exceeded 3 px RMS. Nominal and bright-exposure images remained below it; dim images were unavailable. The 3 px threshold has not been validated on independent or real-camera frames. This diagnostic neither updates calibration nor changes reaching.",
        ORANGE,
        8.1,
    )
    c.showPage()


def _draw_camera_calibration_validation(c: canvas.Canvas, root: Path) -> None:
    _header(c, 15, "Frozen calibration threshold")
    _title(
        c,
        "Labelled calibration shifts improved the threshold",
        "A threshold was selected on seeds 400-404, frozen, then evaluated on fresh seeds 500-504 with broader image noise, brightness, and blur.",
    )

    preview = root / "assets" / "demos" / "camera_calibration_validation.png"
    image = ImageReader(str(preview))
    image_height = CONTENT_W * 548 / 1312
    c.drawImage(
        image,
        MARGIN,
        512,
        width=CONTENT_W,
        height=image_height,
        preserveAspectRatio=True,
        mask="auto",
    )
    _text(
        c,
        MARGIN,
        498,
        "The same held-out nominal frame triggers the lower threshold but not the frozen supervised threshold.",
        7.5,
        MUTED,
    )

    _section_label(c, MARGIN, 475, "Held-out results (available landmark estimates)", BLUE)
    table_bottom = _table(
        c,
        MARGIN,
        458,
        [150, 62, 90, 70, 70, CONTENT_W - 442],
        [
            ["Threshold policy", "RMS px", "Nominal false alarms", "Subtle", "Moderate", "Severe"],
            ["Nominal-only p95", "0.573", "7 / 95 (7.4%)", "100 / 100", "100 / 100", "100 / 100"],
            ["Supervised, 5% cap", "0.841", "0 / 95", "100 / 100", "100 / 100", "100 / 100"],
            ["Fixed exploratory", "3.000", "0 / 95", "60 / 100", "93 / 100", "100 / 100"],
        ],
        font_size=7.1,
        leading=8.5,
    )

    _section_label(c, MARGIN, table_bottom - 22, "Calibration rule", GREEN)
    _paragraph(
        c,
        MARGIN,
        table_bottom - 40,
        "The supervised rule maximized shifted-camera detection on labelled calibration renders while keeping nominal calibration false alarms at or below 5%. It selected 0.841 px, with 0/100 nominal false alarms and 300/300 calibration shifts detected.",
        CONTENT_W,
        8.2,
        11,
        MUTED,
    )

    _callout(
        c,
        MARGIN,
        72,
        CONTENT_W,
        100,
        "Promising result with a narrow validation boundary",
        "On held-out seeds, the supervised rule flagged 0/95 nominal images with visible landmarks and detected all 300 new camera shifts. Five dim probes were unavailable. The study reuses one scene, landmark layout, shift families, and severity ranges; it is not a real-camera or reaching-success result.",
        ORANGE,
        8,
    )
    c.showPage()


def _draw_camera_calibration_transfer(c: canvas.Canvas, root: Path) -> None:
    _header(c, 16, "Frozen threshold transfer")
    _title(
        c,
        "Camera pose changes the residual evidence",
        "The 0.841 px East-view threshold was frozen and tested on 3,000 simulator renders across five camera poses and synthetic radial lens warps.",
    )

    preview = root / "assets" / "demos" / "camera_calibration_transfer.png"
    image = ImageReader(str(preview))
    image_width = CONTENT_W * 0.86
    image_height = image_width * 554 / 1312
    image_x = MARGIN + (CONTENT_W - image_width) / 2
    c.drawImage(
        image,
        image_x,
        513,
        width=image_width,
        height=image_height,
        preserveAspectRatio=True,
        mask="auto",
    )
    _text(
        c,
        MARGIN,
        499,
        "White crosses: expected centers | Green rings: detections | Orange lines: candidate residuals",
        7.3,
        MUTED,
    )

    _section_label(c, MARGIN, 480, "Frozen-threshold camera-shift results", BLUE)
    first_bottom = _table(
        c,
        MARGIN,
        463,
        [100, 82, 68, 78, CONTENT_W - 328],
        [
            ["Pose", "Nominal alarms", "Subtle", "Moderate", "Severe / unavailable"],
            ["East reference", "0/95", "100/100", "100/100", "100/100; 0 unavailable"],
            ["West lower", "2/95", "100/100", "99/99", "82/82; 18 unavailable"],
            ["North higher", "0/95", "99/100", "100/100", "100/100; 0 unavailable"],
            ["Diagonal", "0/95", "100/100", "100/100", "99/99; 1 unavailable"],
            ["East grazing", "Unavailable", "Unavailable", "Unavailable", "Unavailable"],
        ],
        font_size=6.9,
        leading=8.2,
    )

    _section_label(c, MARGIN, first_bottom - 17, "Radial distortion alarms (25 images per coefficient)", GREEN)
    second_bottom = _table(
        c,
        MARGIN,
        first_bottom - 32,
        [CONTENT_W * 0.24, CONTENT_W * 0.25, CONTENT_W * 0.25, CONTENT_W * 0.26],
        [
            ["Synthetic k1", "East", "North higher", "Southwest diagonal"],
            ["-0.005", "0/25", "0/25", "0/25"],
            ["+0.005", "25/25", "0/25", "0/25"],
            ["+0.010", "25/25", "0/25", "1/25"],
            ["+0.020", "25/25", "25/25", "25/25"],
        ],
        font_size=7.0,
        leading=8.3,
    )
    _callout(
        c,
        MARGIN,
        58,
        CONTENT_W,
        second_bottom - 70,
        "A frozen pixel threshold does not transfer uniformly",
        "The same +0.010 radial warp triggered 25/25 East images but 0/25 higher-North images. Edge-clipped expected points produce an unavailable result; their raw candidate RMS is retained. A dense planar follow-up now profiles residuals by radius. Next: validate it on recorded camera frames.",
        ORANGE,
        7.7,
    )
    c.showPage()


def _draw_camera_calibration_planar(c: canvas.Canvas, root: Path) -> None:
    _header(c, 17, "Dense planar-target diagnostic")
    _title(
        c,
        "Residuals now have a radius",
        "A 49-corner planar target adds spatial coverage to the frozen four-marker camera check. The profile is descriptive; the 0.841 px threshold remains specific to the original markers.",
    )
    preview = root / "assets" / "demos" / "camera_calibration_planar.png"
    image = ImageReader(str(preview))
    image_width = CONTENT_W * 0.90
    image_height = image_width * 600 / 960
    image_x = MARGIN + (CONTENT_W - image_width) / 2
    c.drawImage(
        image,
        image_x,
        420,
        width=image_width,
        height=image_height,
        preserveAspectRatio=True,
        mask="auto",
    )
    _text(
        c,
        MARGIN,
        407,
        "Color marks expected image radius; white dots show detected corners; residual vectors are exaggerated 3x.",
        7.0,
        MUTED,
    )
    _section_label(c, MARGIN, 388, "Nominal simulator coverage (five appearance samples per view)", BLUE)
    _table(
        c,
        MARGIN,
        372,
        [112, 92, 91, 92, CONTENT_W - 387],
        [
            ["Pose", "Grid available", "Grid max radius", "4-marker max radius", "Grid median RMS"],
            ["East", "5/5", "241 px", "297 px", "0.28 px"],
            ["West lower", "5/5", "272 px", "343 px", "0.79 px"],
            ["North higher", "5/5", "186 px", "223 px", "0.16 px"],
            ["Southwest diagonal", "4/5", "182 px", "210 px", "0.41 px"],
            ["East grazing", "3/5", "287 px", "366 px", "1.59 px"],
        ],
        font_size=6.8,
        leading=8.1,
    )
    _callout(
        c,
        MARGIN,
        67,
        (CONTENT_W - 12) / 2,
        127,
        "Radius explains a view-dependent miss",
        "East k1=+0.010: median signed radial residual is -0.05 px at 0-80 px and +0.86 px at 240-320 px. The frozen four-marker rule alarms on 5/5 East images, then 0/5 higher-North images at the same synthetic coefficient.",
        GREEN,
        7.6,
    )
    _callout(
        c,
        MARGIN + (CONTENT_W + 12) / 2,
        67,
        (CONTENT_W - 12) / 2,
        127,
        "More field coverage costs visibility",
        "A 0.17 m-square probe reached a 354 px radius but fell to 2/5 West, 4/5 diagonal, and 0/5 grazing availability. The 0.15 m-square target keeps higher availability while undersampling some edge radii. All evidence remains synthetic.",
        ORANGE,
        7.6,
    )
    c.showPage()


def _draw_camera_distortion_results(c: canvas.Canvas) -> None:
    _header(c, 18, "Recorded images and reaching")
    _title(
        c,
        "Distortion correction is conditional",
        "A public-camera fit checks the calibration pipeline; a paired MuJoCo study checks whether optional pixel correction changes reaching.",
    )

    _section_label(c, MARGIN, 674, "Recorded-camera calibration: one OpenCV sequence")
    flow = [
        ("8 training views", "fit focal lengths and radial k1,k2", BLUE),
        ("Freeze intrinsics", "estimate each held-out board pose", GREEN),
        ("4 held-out views", "measure corner reprojection error", ORANGE),
    ]
    box_w = (CONTENT_W - 28) / 3
    for index, (title, body, accent) in enumerate(flow):
        x = MARGIN + index * (box_w + 14)
        _rounded(c, x, 593, box_w, 61, PANEL_ALT)
        _text(c, x + 11, 632, title, 9, accent, "Helvetica-Bold")
        _paragraph(c, x + 11, 614, body, box_w - 22, 7.8, 9.5, MUTED)
        if index < len(flow) - 1:
            _arrow(c, x + box_w + 2, 622, x + box_w + 12, 622, GREEN, 1.1)

    rms_results = [("Zero distortion", 1.445, SUBTLE), ("Radial k1,k2", 0.286, BLUE)]
    bar_x, bar_w = MARGIN + 164, CONTENT_W - 208
    for index, (label, value, color) in enumerate(rms_results):
        y = 566 - index * 22
        _text(c, MARGIN, y + 2, label, 8.2, WHITE)
        _rounded(c, bar_x, y - 2, bar_w, 11, PANEL_ALT, radius=5)
        c.setFillColor(color)
        c.roundRect(bar_x, y - 2, bar_w * value / 1.6, 11, 5, fill=1, stroke=0)
        _text(c, bar_x + bar_w * value / 1.6 + 6, y, f"{value:.3f} px", 8, MUTED)
    _paragraph(
        c,
        MARGIN,
        514,
        "Held-out corners stop within 240 px of the estimated image center. The camera has no published ground-truth intrinsics, so this is a one-sequence fit check, not proof of a better robot camera model.",
        CONTENT_W,
        8,
        10.5,
        MUTED,
    )

    _section_label(c, MARGIN, 476, "MuJoCo reaching: no correction versus exact injected k1")
    legend_y = 456
    c.setFillColor(SUBTLE)
    c.roundRect(MARGIN + 2, legend_y, 9, 9, 2, fill=1, stroke=0)
    _text(c, MARGIN + 16, legend_y + 1, "No correction", 7.5, MUTED)
    c.setFillColor(BLUE)
    c.roundRect(MARGIN + 103, legend_y, 9, 9, 2, fill=1, stroke=0)
    _text(c, MARGIN + 117, legend_y + 1, "Matched coefficient", 7.5, MUTED)

    reaching_rows = [
        ("Open loop | k1 -0.10", 0.84, 1.09),
        ("Open loop | k1 +0.10", 1.48, 1.10),
        ("Image feedback | k1 -0.10", 0.75, 0.68),
        ("Image feedback | k1 +0.10", 0.73, 0.78),
    ]
    chart_x, chart_w = MARGIN + 206, CONTENT_W - 255
    for index, (label, baseline, corrected) in enumerate(reaching_rows):
        group_y = 427 - index * 37
        _text(c, MARGIN, group_y + 4, label, 7.8, WHITE)
        for offset, value, color in ((0, baseline, SUBTLE), (12, corrected, BLUE)):
            y = group_y - offset
            _rounded(c, chart_x, y, chart_w, 8, PANEL_ALT, radius=4)
            c.setFillColor(color)
            c.roundRect(chart_x, y, chart_w * value / 1.65, 8, 4, fill=1, stroke=0)
            _text(c, chart_x + chart_w * value / 1.65 + 5, y + 1, f"{value:.2f} cm", 7.2, MUTED)

    _section_label(c, MARGIN, 281, "20 px noise: paired success changes and error tails", ORANGE)
    _table(
        c,
        MARGIN,
        264,
        [77, 45, 126, 126, CONTENT_W - 374],
        [
            ["Controller", "k1", "No correction\nsuccess; med/P95", "Exact k1\nsuccess; med/P95", "Recovered / lost"],
            ["Open loop", "-0.10", "28/100; 6.46/13.98 cm", "29/100; 6.65/13.94 cm", "3 / 2"],
            ["Open loop", "+0.10", "31/100; 6.39/13.81 cm", "33/100; 6.39/13.86 cm", "2 / 0"],
            ["Image feedback", "-0.10", "81/100; 2.67/6.18 cm", "79/100; 2.78/6.26 cm", "6 / 8"],
            ["Image feedback", "+0.10", "83/100; 2.75/6.35 cm", "82/100; 2.29/6.86 cm", "7 / 8"],
        ],
        font_size=6.5,
        leading=8.0,
    )
    _callout(
        c,
        MARGIN,
        45,
        CONTENT_W,
        80,
        "Correction did not fix the high-noise failure",
        "At 20 px, recovered and lost success counts were close; feedback lost one or two successes. Positive k1 feedback improved its median but worsened P95 by 0.51 cm. The 2 px chart above has 100/100 successes in every row. Exact coefficients are an oracle check, +/-0.10 is a simulation stress value, and fresh visual confirmation was not evaluated.",
        ORANGE,
        7.5,
    )
    c.showPage()


def _draw_next_steps(c: canvas.Canvas) -> None:
    _header(c, 19, "Held-out motion envelope")
    _title(
        c,
        "A strict envelope cuts stale positives, with low coverage",
        "A training-calibrated image margin and target-speed cap reduce delivery mismatches, but this held-out result does not justify a runtime rule.",
    )
    cards = [
        ("2.54 cm", "empirical q99 margin", BLUE),
        ("8.2%", "envelope coverage", GREEN),
        ("0 / 2,340", "false confirmations", ORANGE),
        ("23 / 16,168", "delivery-event misses", RED),
    ]
    card_w = (CONTENT_W - 18) / 4
    for index, (value, label, accent) in enumerate(cards):
        x = MARGIN + index * (card_w + 6)
        _rounded(c, x, 651, card_w, 57, PANEL)
        _text(c, x + 10, 684, value, 14, accent, "Helvetica-Bold")
        _paragraph(c, x + 10, 666, label, card_w - 18, 7.1, 8.5, MUTED)

    _section_label(c, MARGIN, 624, "Fully delayed RGB | two-observation control delay", BLUE)
    _text(c, MARGIN + 133, 600, "Confirmation coverage", 7.6, MUTED, "Helvetica-Bold")
    _text(c, MARGIN + 365, 600, "False at delivery (0-10%)", 7.2, MUTED, "Helvetica-Bold")
    plot_rows = [
        ("Age guard <= 1.0 s", 334 / 468, 23 / 468, BLUE),
        ("Envelope <= 0.126 m/s", 20 / 468, 0 / 468, GREEN),
    ]
    for index, (label, coverage, false_rate, accent) in enumerate(plot_rows):
        y = 567 - 34 * index
        _text(c, MARGIN, y + 4, label, 7.1, WHITE, "Helvetica-Bold")
        _rounded(c, MARGIN + 133, y, 136, 9, PANEL_ALT, radius=4)
        c.setFillColor(accent)
        c.roundRect(MARGIN + 133, y, 136 * coverage, 9, 4, fill=1, stroke=0)
        _text(c, MARGIN + 273, y + 1, f"{100 * coverage:.1f}%", 7.3, MUTED)
        _rounded(c, MARGIN + 365, y, 52, 9, PANEL_ALT, radius=4)
        c.setFillColor(accent)
        c.roundRect(MARGIN + 365, y, 52 * false_rate / 0.10, 9, 4, fill=1, stroke=0)
        _text(c, MARGIN + 423, y + 1, f"{100 * false_rate:.1f}%", 7.3, MUTED)
        _text(c, MARGIN + 133, y - 10, f"{round(coverage * 468):,}/468", 6.4, SUBTLE)
        _text(c, MARGIN + 365, y - 10, f"{round(false_rate * 468)}/468", 6.4, SUBTLE)

    _callout(
        c,
        MARGIN,
        438,
        CONTENT_W,
        70,
        "The empirical margin is not a guarantee",
        "Across the three profile-specific delivery streams, the q99 image margin underbounded capture distance on 202/16,168 events and the motion envelope exceeded delivery distance on 23/16,168. The streams replay the same captures with different loss and age, so records are correlated. Two-frame confirmation made no false decisions here, but event-level violations remain.",
        ORANGE,
        7.4,
    )
    mode_rows = [
        ["Control input", "Age guard: confirms; false", "Envelope: confirms; false", "Physical reach at stop"],
        ["Fresh RGB", "412/468; 33", "49/468; 0", "137/156"],
        ["Fully delayed RGB, 1 obs", "387/468; 52", "25/468; 0", "91/156"],
        ["Fully delayed RGB, 2 obs", "334/468; 23", "20/468; 0", "90/156"],
        ["Delayed target + current tip, 1 obs", "413/468; 38", "40/468; 0", "117/156"],
        ["Delayed target + current tip, 2 obs", "403/468; 39", "58/468; 0", "125/156"],
    ]
    _table(
        c,
        MARGIN,
        418,
        [163, 117, 117, CONTENT_W - 397],
        mode_rows,
        font_size=6.3,
        leading=7.4,
    )
    _paragraph(
        c,
        MARGIN,
        257,
        "Each camera-policy cell pools three delivery profiles (n=468). Physical reaching is scored at controller stop over 156 trials per mode; the later camera decision remains a separate result.",
        CONTENT_W,
        7.2,
        9.0,
        MUTED,
    )
    _callout(
        c,
        MARGIN,
        132,
        CONTENT_W,
        83,
        "Why calibrate image uncertainty conditionally?",
        "A single 2.54 cm q99 margin reduced held-out coverage to 8.2%. Pixel location changes how camera rays map to the table, while marker-localization quality varies across frames. The next page reports the held-out test of those image-only signals. Completion policies stay offline.",
        BLUE,
        7.8,
    )
    _section_label(c, MARGIN, 108, "Reproduce the benchmark and paired demo", GREEN)
    _text(c, MARGIN, 88, "uv run visual-servo-motion-envelope-benchmark", 6.5, MUTED, "Courier")
    _text(c, MARGIN, 71, "uv run --extra demo python scripts/generate_motion_envelope_demo.py", 6.5, MUTED, "Courier")
    _text(c, MARGIN, 49, "No runtime controller change; image confirmation and controller-stop physical reaching remain separate.", 7.2, ORANGE)
    c.showPage()


def _draw_conditional_uncertainty(c: canvas.Canvas, repository: Path) -> None:
    import json

    report_path = repository / "artifacts" / "conditional-uncertainty-benchmark.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report["results"]
    selected = [
        row
        for row in summary
        if row["control_mode"] == "fully_delayed_rgb"
        and row["control_delay_observations"] == 2
    ]
    policies = (
        ("two_post_settle_age_1s", "Age guard <= 1.0 s", BLUE),
        ("two_post_settle_prior_global_q99", "Prior global q99", ORANGE),
        ("two_post_settle_new_global_q99", "New global q99", PURPLE),
        ("two_post_settle_conditional_q99", "Conditional q99", GREEN),
    )
    totals = {}
    for policy, label, accent in policies:
        rows = [row for row in selected if row["policy"] == policy]
        trials = sum(int(row["trials"]) for row in rows)
        confirmed = sum(int(row["visually_confirmed"]) for row in rows)
        false = sum(int(row["false_confirmations_at_delivery"]) for row in rows)
        totals[policy] = (label, accent, trials, confirmed, false)

    _header(c, 20, "Conditional image-distance uncertainty")
    _title(
        c,
        "The held-out comparison keeps the decision evidence separate",
        "A three-band empirical q99 uses only pixel-derived uncertainty features at decision time; simulator distances score coverage after the fact.",
    )
    bins = report["calibration"]["bins"]
    card_w = (CONTENT_W - 18) / 4
    values = [
        (f"{report['calibration']['global_q99_margin_m'] * 100:.2f} cm", "new global q99", BLUE),
        (f"{len(bins)}", "conditional feature bands", GREEN),
        (f"{report['protocol']['calibration_frame_count']:,}", "training camera frames", PURPLE),
        (f"{len(report['protocol']['validation_seeds'])}", "held-out seeds", ORANGE),
    ]
    for index, (value, label, accent) in enumerate(values):
        x = MARGIN + index * (card_w + 6)
        _rounded(c, x, 651, card_w, 57, PANEL)
        _text(c, x + 10, 684, value, 14, accent, "Helvetica-Bold")
        _paragraph(c, x + 10, 666, label, card_w - 18, 7.1, 8.5, MUTED)

    _section_label(c, MARGIN, 624, "Fully delayed RGB | two-observation control delay", BLUE)
    _text(c, MARGIN, 598, "Policy", 7.4, MUTED, "Helvetica-Bold")
    _text(c, MARGIN + 168, 598, "Confirmation coverage", 7.4, MUTED, "Helvetica-Bold")
    _text(c, MARGIN + 349, 598, "False at delivery", 7.4, MUTED, "Helvetica-Bold")
    max_trials = max((row[2] for row in totals.values()), default=1)
    for index, (policy, label, accent) in enumerate(policies):
        label, accent, trials, confirmed, false = totals[policy]
        y = 571 - index * 37
        _text(c, MARGIN, y + 4, label, 7.1, WHITE, "Helvetica-Bold")
        _rounded(c, MARGIN + 168, y, 112, 9, PANEL_ALT, radius=4)
        c.setFillColor(accent)
        c.roundRect(MARGIN + 168, y, 112 * confirmed / trials if trials else 0, 9, 4, fill=1, stroke=0)
        _text(c, MARGIN + 286, y + 1, f"{100 * confirmed / trials if trials else 0:.1f}% ({confirmed}/{trials})", 7.1, MUTED)
        _rounded(c, MARGIN + 349, y, 48, 9, PANEL_ALT, radius=4)
        c.setFillColor(accent)
        c.roundRect(MARGIN + 349, y, 48 * false / trials if trials else 0, 9, 4, fill=1, stroke=0)
        _text(c, MARGIN + 403, y + 1, f"{100 * false / trials if trials else 0:.1f}% ({false}/{trials})", 7.1, MUTED)

    _callout(
        c,
        MARGIN,
        335,
        CONTENT_W,
        83,
        "Conditional q99 margins by increasing image-noise feature",
        "".join(
            f"Band {item['index'] + 1}: {item['margin_m'] * 100:.2f} cm from {item['sample_count']:,} training frames; "
            for item in bins
        )
        + "The feature combines the detected pixel centers, local projection sensitivity, 2 px configured noise, and the minimum heuristic marker-quality score. The score is not a probability. Underpopulated bands use the pooled training q99.",
        GREEN,
        7.1,
    )

    stop_records = {}
    for record in report["validation_trial_results"]:
        if record["control_mode"] != "fully_delayed_rgb" or record["control_delay_observations"] != 2:
            continue
        key = (record["motion_condition"], record["seed"], record["episode"], record["target_motion_phase_rad"])
        stop_records[key] = bool(record["physical_success_at_controller_stop"])
    stop_count = sum(stop_records.values())
    stop_total = len(stop_records)
    diagnostic_count = sum(int(row["delivered_post_settle_frame_events"]) for row in report["validation_conditional_diagnostics"])
    diagnostic_violations = sum(int(row["image_margin_violations"]) for row in report["validation_conditional_diagnostics"])
    _paragraph(
        c,
        MARGIN,
        285,
        f"Physical reaching at controller stop: {stop_count}/{stop_total}. Conditional image-margin undercoverage: {diagnostic_violations}/{diagnostic_count} profile-specific delivered post-settle frame events; those events reuse camera captures across transport profiles. Neither metric is the policy's fresh visual-confirmation count.",
        CONTENT_W,
        7.1,
        9.0,
        MUTED,
    )
    _section_label(c, MARGIN, 245, "Read the evidence and replay the paired demo", GREEN)
    _text(c, MARGIN, 224, "uv run visual-servo-conditional-uncertainty-benchmark", 6.7, MUTED, "Courier")
    _text(c, MARGIN, 207, "uv run --extra demo python scripts/generate_conditional_uncertainty_demo.py", 6.7, MUTED, "Courier")
    _text(c, MARGIN, 177, "Image measurements choose the margin; simulation labels grade the decision.", 7.4, ORANGE, "Helvetica-Bold")
    _text(c, MARGIN, 158, "All completion rules remain offline experiments. The fully delayed RGB rows delay target and tip together; they are not the current-joint-state hybrid.", 6.9, MUTED)
    _callout(
        c,
        MARGIN,
        63,
        CONTENT_W,
        73,
        "Maintainer-facing takeaway",
        "A conditional margin is useful only if the held-out table shows a better coverage/false-confirmation trade-off without breaking the measured frame-level envelope. Keep the prior global q99 result visible even if the new calibration improves it.",
        BLUE,
        7.4,
    )
    c.showPage()


def _draw_perception_candidate_validation(c: canvas.Canvas, repository: Path) -> None:
    report_path = repository / "results" / "perception-candidate-validation-summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["condition_summaries"]
    total_trials = int(report["trial_count"])
    outcomes = (
        ("strict_successes", "#57D6A3"),
        ("imprecise_localizations", "#FFB86C"),
        ("wrong_targets", "#FF7B72"),
        ("abstentions", "#718096"),
    )
    pooled = {}
    for prefix in ("baseline", "candidate"):
        pooled[prefix] = {
            key: sum(int(row[f"{prefix}_{key}"]) for row in rows)
            for key, _ in outcomes
        }

    _header(c, 21, "Held-out target detection")
    _title(
        c,
        "A target-size prior helps, but identity still fails",
        "An opt-in circle detector is compared with the unchanged largest-red-contour baseline on disjoint rendered lighting, distractor size and distance, and occlusion-width ranges.",
    )
    _section_label(c, MARGIN, 665, "Paired detector outcomes across 450 held-out frames", ORANGE)
    bar_x = MARGIN + 107
    bar_w = CONTENT_W - 107
    for index, (prefix, label) in enumerate(
        (("baseline", "Default"), ("candidate", "Circle prior"))
    ):
        y = 615 - index * 58
        _text(c, MARGIN, y + 4, label, 9, WHITE, "Helvetica-Bold")
        cursor = bar_x
        for key, color in outcomes:
            count = pooled[prefix][key]
            width = bar_w * count / total_trials
            if width > 0:
                c.setFillColor(HexColor(color))
                c.rect(cursor, y, width, 18, fill=1, stroke=0)
                cursor += width
        c.setStrokeColor(GRID)
        c.setLineWidth(0.55)
        c.rect(bar_x, y, bar_w, 18, fill=0, stroke=1)
        strict = pooled[prefix]["strict_successes"]
        wrong = pooled[prefix]["wrong_targets"]
        abstain = pooled[prefix]["abstentions"]
        _text(
            c,
            bar_x,
            y - 13,
            f"Within 4 px: {strict}/{total_trials}   |   Wrong target: {wrong}/{total_trials}   |   Abstained: {abstain}/{total_trials}",
            7.5,
            MUTED,
        )
    legend_x = MARGIN + 8
    legend_y = 490
    legend = (
        ("Within 4 px", GREEN),
        ("Imprecise, within target radius", ORANGE),
        ("Wrong target", RED),
        ("Abstained", SUBTLE),
    )
    for label, color in legend:
        c.setFillColor(color)
        c.roundRect(legend_x, legend_y - 1, 8, 8, 2, fill=1, stroke=0)
        _text(c, legend_x + 13, legend_y, label, 7.1, MUTED)
        legend_x += 116 if label != "Imprecise, within target radius" else 180

    _section_label(c, MARGIN, 462, "The red-disk condition remains the hard case", BLUE)
    breakdown = [["Clutter", "Default: strict / wrong / abstain", "Circle prior: strict / wrong / abstain"]]
    for clutter in ("none", "red_disk", "red_box"):
        condition_rows = [row for row in rows if row["clutter"] == clutter]
        values = [clutter.replace("_", " ")]
        for prefix in ("baseline", "candidate"):
            strict = sum(int(row[f"{prefix}_strict_successes"]) for row in condition_rows)
            wrong = sum(int(row[f"{prefix}_wrong_targets"]) for row in condition_rows)
            abstain = sum(int(row[f"{prefix}_abstentions"]) for row in condition_rows)
            values.append(f"{strict}/150   |   {wrong}/150   |   {abstain}/150")
        breakdown.append(values)
    table_end = _table(
        c,
        MARGIN,
        446,
        [82, 209, CONTENT_W - 291],
        breakdown,
        font_size=7.1,
        leading=8.5,
    )

    _callout(
        c,
        MARGIN,
        249,
        CONTENT_W,
        max(85, table_end - 273),
        "What improved, and what remains unresolved",
        "The circle prior strictly localized 175/450 frames versus 8/450 for the default, and reduced wrong-target outputs from 136 to 6. It abstained on 266 frames. All six false accepts were small red disks under occlusion; in some frames the candidate's size-fit score was high even though it selected the distractor. The detector needs an identity cue such as a distinct marker, temporal tracking, or depth before it can guide control.",
        ORANGE,
        8.1,
    )
    _callout(
        c,
        MARGIN,
        143,
        CONTENT_W,
        82,
        "Trace the implementation",
        "perception_candidate.py computes the 9 cm marker's projected radius, runs red-mask Hough circles, and abstains on competing circles. perception_candidate_validation.py randomizes scene factors and compares both detectors on identical frames. The default detector is unchanged; the experiment does not run the arm.",
        GREEN,
        7.8,
    )
    _section_label(c, MARGIN, 116, "Replay the validation", GREEN)
    _text(
        c,
        MARGIN,
        96,
        "uv run --locked visual-servo-perception-candidate-validation --base-seed 9400",
        6.3,
        MUTED,
        "Courier",
    )
    _text(
        c,
        MARGIN,
        79,
        "450 localization trials only | physical reaching and fresh visual confirmation not measured",
        7.2,
        ORANGE,
    )
    c.showPage()


def main() -> Path:
    repository = Path(__file__).resolve().parents[1]
    output = repository / "output" / "pdf" / "visual-servo-mujoco-project-explainer.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
    c.setTitle("Visual Servo MuJoCo Project Explainer")
    c.setAuthor("Jad")
    c.setSubject("Visual servoing project architecture, code, benchmarks, and next milestone")
    _draw_cover(c, repository)
    _draw_pipeline(c)
    _draw_control(c)
    _draw_code_map(c)
    _draw_benchmarks(c)
    _draw_sensor_timing(c)
    _draw_timestamped_verification(c, repository)
    _draw_completion_speed(c, repository)
    _draw_negative_results(c)
    _draw_verifier(c)
    _draw_verifier_robustness(c)
    _draw_verifier_quality(c)
    _draw_frozen_quality_validation(c)
    _draw_camera_calibration_health(c, repository)
    _draw_camera_calibration_validation(c, repository)
    _draw_camera_calibration_transfer(c, repository)
    _draw_camera_calibration_planar(c, repository)
    _draw_camera_distortion_results(c)
    _draw_next_steps(c)
    _draw_conditional_uncertainty(c, repository)
    _draw_perception_candidate_validation(c, repository)
    c.save()
    print(f"Wrote {output}")
    return output


if __name__ == "__main__":
    main()
