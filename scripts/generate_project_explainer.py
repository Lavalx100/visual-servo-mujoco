"""Create the illustrated portfolio project explainer PDF with ReportLab."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.colors import Color, HexColor
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
    _text(c, MARGIN, PAGE_H - 190, "From planar servoing to multi-arm contact", 12, MUTED)
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
    _text(c, MARGIN, 48, "Local work on feat/cooperative-multiarm-pick-demo | Updated 22 Oct 2026", 8, MUTED)
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


def _draw_multiarm_integration(c: canvas.Canvas, root: Path) -> None:
    report = json.loads(
        (root / "results" / "cooperative-pick-pad-size-comparison.json").read_text()
    )
    baseline = {row["controller"]: row for row in report["baseline"]["conditions"]}
    candidate = {row["controller"]: row for row in report["candidate"]["conditions"]}

    _header(c, 22, "Multi-arm integration")
    _title(
        c,
        "Connecting perception to physical contact",
        "The tri-arm extension reuses the project's detector and camera geometry, then evaluates a real MuJoCo contact task with the same free-payload scene as the scripted baseline.",
    )

    nodes = [
        ("01  RENDER", "Top-down MuJoCo RGB; a visual-only red marker sits above the free payload.", BLUE),
        ("02  DETECT + PROJECT", "Unchanged detect_red_target() and pixel_to_table_xy() produce target XY.", GREEN),
        ("03  PAIRED INPUT", "Camera estimate plans the pick. Oracle XY uses ground truth only as a diagnostic.", ORANGE),
        ("04  PLAN THREE ARMS", "Base yaw + elbow-up 3R IK + gripper yaw reuse the shared pick phases.", BLUE),
        ("05  CONTACT + SCORE", "Free payload, gravity, friction, and arm colliders drive lift and physical checks.", PURPLE),
    ]
    box_w = 94
    box_h = 110
    gap = (CONTENT_W - box_w * len(nodes)) / (len(nodes) - 1)
    top = 654
    for index, (label, body, accent) in enumerate(nodes):
        x = MARGIN + index * (box_w + gap)
        _rounded(c, x, top - box_h, box_w, box_h, PANEL_ALT, stroke=GRID, radius=8)
        c.setFillColor(accent)
        c.roundRect(x, top - 7, box_w, 7, 3, fill=1, stroke=0)
        _paragraph(c, x + 8, top - 25, label, box_w - 16, 7.0, 8, accent, "Helvetica-Bold")
        _paragraph(c, x + 8, top - 47, body, box_w - 16, 7.1, 9.5, MUTED)
        if index < len(nodes) - 1:
            _arrow(c, x + box_w + 2, top - box_h / 2, x + box_w + gap - 2, top - box_h / 2, GREEN, 1.0)

    _section_label(c, MARGIN, 519, "Follow the shared code path", GREEN)
    code_rows = [
        ("run.py", "detect_red_target", "Existing OpenCV HSV detector returns the red marker center."),
        ("controller.py", "pixel_to_table_xy", "Existing pinhole projection maps that pixel to tabletop XY."),
        ("camera_guided_cooperative_pick.py", "run_trial / _set_camera_guided_arm_controls", "Pairs camera/oracle input and updates the three arm references."),
        ("cooperative_pick_demo.py", "_arm_collision_masks / desired_pick_state", "Reuses cross-arm colliders and task phases; visual wrist adapters, palm rails, and finger shanks connect to pads, which remain the grasp surfaces."),
    ]
    row_y = 497
    for index, (filename, function, detail) in enumerate(code_rows):
        row_h = 48
        _rounded(c, MARGIN, row_y - row_h, CONTENT_W, row_h, PANEL if index % 2 == 0 else PANEL_ALT, radius=7)
        _text(c, MARGIN + 11, row_y - 15, filename, 7.6, BLUE, "Helvetica-Bold")
        _right_text(c, PAGE_W - MARGIN - 11, row_y - 15, function, 6.8, GREEN, "Helvetica-Bold")
        _paragraph(c, MARGIN + 11, row_y - 30, detail, CONTENT_W - 22, 7.3, 9, MUTED)
        row_y -= row_h + 4

    _section_label(c, MARGIN, 277, "Same-pose held-out contact-geometry comparison", ORANGE)
    chart_x = MARGIN + 22
    chart_w = CONTENT_W - 120
    chart_bottom = 129
    for condition_index, (key, label) in enumerate(
        (("camera_estimate", "Camera estimate"), ("oracle_xy", "Oracle XY diagnostic"))
    ):
        y = chart_bottom + condition_index * 64
        _text(c, MARGIN, y + 68, label, 7.3, WHITE, "Helvetica-Bold")
        for pad_index, row in enumerate((baseline[key], candidate[key])):
            bar_y = y + 24 + pad_index * 18
            width = chart_w * row["physical_stable_lift_success_rate"]
            color = BLUE if pad_index == 0 else GREEN
            _text(c, chart_x, bar_y + 3, f"{row['pad_half_size_m'] * 1000:.0f} mm", 6.9, MUTED)
            x = chart_x + 34
            c.setFillColor(PANEL_ALT)
            c.roundRect(x, bar_y, chart_w - 34, 12, 3, fill=1, stroke=0)
            c.setFillColor(color)
            c.roundRect(x, bar_y, max(2, width), 12, 3, fill=1, stroke=0)
            _right_text(
                c,
                PAGE_W - MARGIN,
                bar_y + 2,
                f"{row['physical_stable_lift_successes']}/{row['trials']}  ({row['physical_stable_lift_success_rate']:.0%})",
                7.0,
                WHITE,
                "Helvetica-Bold",
            )

    _callout(
        c,
        MARGIN,
        56,
        CONTENT_W,
        78,
        "What the result points to next",
        "On the same 30 poses, 16 mm pads lifted in 12/30 camera trials versus 4/30 at 14 mm; detector median error stayed 0.48 mm. Yet arm-environment and nonfinger payload contacts remain common. Improve collision-aware approach and grasp reliability before changing perception. This is one-shot camera-guided planning: physical lifting is not fresh visual confirmation, and the camera is not used during the lift.",
        ORANGE,
        7.7,
    )
    c.showPage()


def _draw_release_profile(c: canvas.Canvas, root: Path) -> None:
    report = json.loads(
        (root / "results" / "cooperative-pick-release-heldout-comparison.json").read_text()
    )
    outcomes = report["outcomes_by_condition"]
    _header(c, 23, "Release collision study")
    _title(
        c,
        "A safer release follows the measured failure",
        "Contact attribution showed that many successful holds became strict failures during release and withdrawal. The clearance profile changes only those actuator targets, preserving the free payload and active colliders.",
    )

    nodes = [
        ("1  IDENTIFY", "Wrist-to-payload and fingertip-to-table contacts cluster during release and withdrawal.", ORANGE),
        ("2  CLEAR", "Raise the lowest open tool offset to 35 mm and keep tool centers at least 110 mm from the target.", GREEN),
        ("3  REPLAY", "Gravity, friction, arm colliders, and the strict full-cycle rule remain active.", BLUE),
    ]
    gap = 10
    box_w = (CONTENT_W - gap * (len(nodes) - 1)) / len(nodes)
    for index, (label, body, color) in enumerate(nodes):
        x = MARGIN + index * (box_w + gap)
        _rounded(c, x, 622, box_w, 78, PANEL_ALT, stroke=GRID, radius=8)
        _text(c, x + 11, 681, label, 8, color, "Helvetica-Bold")
        _paragraph(c, x + 11, 661, body, box_w - 22, 7.4, 10, MUTED)
        if index < len(nodes) - 1:
            _arrow(c, x + box_w + 1, 659, x + box_w + gap - 1, 659, GREEN, 1.0)

    _section_label(c, MARGIN, 588, "Fresh paired validation | seed 38200 | 30 poses", GREEN)
    chart_x = MARGIN + 157
    chart_w = 270
    def draw_pair(y: float, title: str, metric: str, condition: str) -> None:
        base = outcomes[condition]["baseline"][metric]["successes"]
        candidate = outcomes[condition]["clearance"][metric]["successes"]
        _text(c, MARGIN, y + 6, title, 7.7, WHITE, "Helvetica-Bold")
        for index, (name, value, color) in enumerate(
            (("base", base, BLUE), ("clear", candidate, GREEN))
        ):
            bar_y = y - 11 - index * 18
            _text(c, chart_x - 41, bar_y + 2, name, 6.5, MUTED)
            c.setFillColor(PANEL_ALT)
            c.roundRect(chart_x, bar_y, chart_w, 10, 3, fill=1, stroke=0)
            c.setFillColor(color)
            c.roundRect(chart_x, bar_y, chart_w * value / 30, 10, 3, fill=1, stroke=0)
            _right_text(c, PAGE_W - MARGIN, bar_y + 1, f"{value}/30", 7, WHITE, "Helvetica-Bold")

    draw_pair(560, "Camera strict score", "physical_stable_lift_success", "camera_estimate")
    draw_pair(490, "Oracle strict score", "physical_stable_lift_success", "oracle_xy")
    _paragraph(
        c,
        MARGIN,
        424,
        "Payload-hold counts did not change: camera 26/30, oracle 23/30. The strict-score gains were paired: 12 camera poses and 12 oracle poses passed only under clearance; none passed only under baseline.",
        CONTENT_W,
        8.0,
        11,
        MUTED,
    )

    _section_label(c, MARGIN, 384, "Unexpected contacts by trial", ORANGE)
    contact_text = (
        "Camera estimate: arm-table 7 -> 2; nonfinger arm-payload 14 -> 4.  "
        "Oracle XY: arm-table 6 -> 0; nonfinger arm-payload 18 -> 5.\n"
        "Counts include any contact over the full trial. Remaining failures cluster in earlier grasp, lift, and lowering phases."
    )
    _callout(c, MARGIN, 266, CONTENT_W, 93, "What changed in the simulator", contact_text, ORANGE, 8.0)

    _callout(
        c,
        MARGIN,
        113,
        CONTENT_W,
        124,
        "Interpretation and next work",
        "The 35 mm and 110 mm settings were tuned after reviewing seed 38100; seed 38200 is a separate evaluation set, still only 30 poses. The selected demo pose is already in that aggregate. This measures physical contact and lift only: no new image confirms the payload after release. Broaden the evaluation and address remaining approach/lift contacts before claiming a generally reliable grasp.",
        GREEN,
        8.0,
    )
    _text(
        c,
        MARGIN,
        83,
        "Code: _apply_release_profile() and run_trial() | Evidence: results/cooperative-pick-release-heldout-comparison.json",
        6.8,
        MUTED,
        "Courier",
    )
    c.showPage()


def _draw_approach_clearance(c: canvas.Canvas, root: Path) -> None:
    report = json.loads(
        (root / "results" / "cooperative-pick-approach-clearance-comparison.json").read_text()
    )
    outcomes = report["outcomes_by_condition"]
    paired = report["paired_outcomes"]
    baseline_mm = round(report["baseline"]["lowest_approach_tool_height_offset_m"] * 1000)
    candidate_mm = round(report["candidate"]["lowest_approach_tool_height_offset_m"] * 1000)
    _header(c, 24, "Approach-clearance study")
    _title(
        c,
        "A 15 mm lift clears the lowest open pad",
        "Contact traces revealed a table strike during the approach. Raising only that arm's open approach target removes the modeled overlap while leaving the finger geometry and closed grasp target unchanged.",
    )

    _section_label(c, MARGIN, 690, "Read the cross-section")
    panel_gap = 12
    panel_width = (CONTENT_W - panel_gap) / 2
    panel_y, panel_h = 450, 226
    table_y = 488
    payload_center_y = table_y + 60
    scale = 1.0
    for index, (title, offset_mm, candidate) in enumerate(
        (("5 mm baseline", baseline_mm, False), ("20 mm candidate", candidate_mm, True))
    ):
        x = MARGIN + index * (panel_width + panel_gap)
        _rounded(c, x, panel_y, panel_width, panel_h, PANEL, stroke=GRID, radius=10)
        _text(c, x + 12, panel_y + panel_h - 22, title, 9.5, GREEN if candidate else BLUE, "Helvetica-Bold")
        _text(c, x + 12, panel_y + panel_h - 38, f"tool center = payload center + {offset_mm} mm", 7.2, MUTED)
        # Cross-section uses the actual local pad offset: its center is 50 mm
        # below tool Z and its half-height is 16 mm.
        tool_y = payload_center_y + offset_mm * scale
        pad_center_y = payload_center_y + (offset_mm - 50) * scale
        pad_bottom_y = pad_center_y - 16 * scale
        c.setStrokeColor(Color(BLUE.red, BLUE.green, BLUE.blue, alpha=0.65))
        c.setDash(2, 2)
        c.line(x + 21, payload_center_y, x + panel_width - 16, payload_center_y)
        c.setDash()
        _rounded(c, x + 43, table_y, 46, 120, PANEL_ALT, stroke=HexColor("#496b8a"), radius=8)
        _text(c, x + 50, payload_center_y - 3, "PAYLOAD", 5.7, MUTED, "Helvetica-Bold")
        c.setStrokeColor(HexColor("#8D9AA8"))
        c.setLineWidth(4)
        c.line(x + 18, table_y, x + panel_width - 16, table_y)
        c.setFillColor(PANEL_ALT)
        c.rect(x + 18, table_y - 18, panel_width - 34, 17, fill=1, stroke=0)
        _text(c, x + 21, table_y - 13, "TABLE TOP  |  z = 0.180 m", 5.8, MUTED, "Helvetica-Bold")
        c.setFillColor(BLUE)
        c.circle(x + 165, tool_y, 4, fill=1, stroke=0)
        c.setStrokeColor(BLUE)
        c.setLineWidth(2.3)
        c.line(x + 165, tool_y - 4, x + 165, pad_center_y + 16)
        c.setFillColor(HexColor("#CED9E5"))
        c.setStrokeColor(WHITE)
        c.setLineWidth(0.8)
        c.roundRect(x + 149, pad_center_y - 16, 32, 32, 4, fill=1, stroke=1)
        _text(c, x + 105, tool_y + 2, "tool center", 6.0, MUTED)
        _text(c, x + 187, pad_center_y - 2, "PAD", 6.0, WHITE, "Helvetica-Bold")
        if candidate:
            c.setStrokeColor(GREEN)
            c.setLineWidth(1.4)
            c.line(x + 198, table_y, x + 198, pad_bottom_y)
            c.line(x + 194, table_y, x + 202, table_y)
            c.line(x + 194, pad_bottom_y, x + 202, pad_bottom_y)
            _text(c, x + 204, (table_y + pad_bottom_y) / 2 - 2, "+14 mm", 7.4, GREEN, "Helvetica-Bold")
            status, status_color = "14 mm TABLE CLEARANCE", GREEN
        else:
            c.setStrokeColor(RED)
            c.setLineWidth(2)
            c.line(x + 191, table_y - 5, x + 201, table_y + 5)
            c.line(x + 191, table_y + 5, x + 201, table_y - 5)
            status, status_color = "1 mm NOMINAL TABLE OVERLAP", RED
        _text(c, x + 12, panel_y + 10, status, 7.2, status_color, "Helvetica-Bold")

    _section_label(c, MARGIN, 426, "Exact paired held-out score | seed 38300")
    chart_y, chart_h = 282, 132
    _rounded(c, MARGIN, chart_y, CONTENT_W, chart_h, PANEL, stroke=GRID, radius=10)
    sub_gap = 16
    sub_width = (CONTENT_W - 3 * sub_gap) / 2
    for index, (condition, label) in enumerate(
        (("camera_estimate", "Camera estimate"), ("oracle_xy", "Oracle XY diagnostic"))
    ):
        x = MARGIN + sub_gap + index * (sub_width + sub_gap)
        base = outcomes[condition]["baseline"]["physical_stable_lift_success"]
        candidate = outcomes[condition]["candidate"]["physical_stable_lift_success"]
        pair = paired[condition]["physical_stable_lift_success"]
        _text(c, x, chart_y + chart_h - 21, label, 8.5, WHITE, "Helvetica-Bold")
        for row_index, (name, value, color) in enumerate(
            ((f"{baseline_mm} mm", base, BLUE), (f"{candidate_mm} mm", candidate, GREEN))
        ):
            bar_y = chart_y + chart_h - 48 - row_index * 20
            _text(c, x, bar_y + 2, name, 6.5, MUTED)
            bar_x = x + 40
            bar_w = sub_width - 92
            c.setFillColor(PANEL_ALT)
            c.roundRect(bar_x, bar_y, bar_w, 10, 3, fill=1, stroke=0)
            c.setFillColor(color)
            c.roundRect(bar_x, bar_y, max(2, bar_w * value["rate"]), 10, 3, fill=1, stroke=0)
            _right_text(c, x + sub_width - 4, bar_y + 1, f"{value['successes']}/30", 7, WHITE, "Helvetica-Bold")
        ci = base["wilson_95_interval"], candidate["wilson_95_interval"]
        _text(c, x, chart_y + 30, f"Wilson 95%: {ci[0][0]:.0%}-{ci[0][1]:.0%} / {ci[1][0]:.0%}-{ci[1][1]:.0%}", 6.1, MUTED)
        _text(c, x, chart_y + 14, f"Paired: {pair['candidate_only_success']} candidate-only, {pair['baseline_only_success']} baseline-only", 6.1, GREEN if pair["candidate_only_success"] >= pair["baseline_only_success"] else ORANGE)

    _section_label(c, MARGIN, 263, "Trace the result and keep the limits in view", ORANGE)
    cards_y, cards_h = 72, 178
    card_gap = 9
    card_w = (CONTENT_W - 2 * card_gap) / 3
    cards = [
        (
            "The paired failure",
            "Pose 38301-003: arm0_right_pad contacts the table during lower-to-grasp and finger closure. The 5 mm run makes no stable grasp; the 20 mm replay passes the same physical score.",
            ORANGE,
        ),
        (
            "What changed",
            "Camera score: 27/30 to 30/30, with three candidate-only wins and no baseline-only wins. Any camera-trial arm-table/ground contact falls from 2/30 to 0/30.",
            GREEN,
        ),
        (
            "What remains uncertain",
            "Oracle score moves 28/30 to 27/30. Camera confidence intervals overlap; n=30, ideal fixed calibration, no hardware, and no fresh post-pick visual confirmation.",
            BLUE,
        ),
    ]
    for index, (heading, body, color) in enumerate(cards):
        x = MARGIN + index * (card_w + card_gap)
        _rounded(c, x, cards_y, card_w, cards_h, PANEL_ALT, stroke=GRID, radius=8)
        _text(c, x + 10, cards_y + cards_h - 21, heading, 8.2, color, "Helvetica-Bold")
        _paragraph(c, x + 10, cards_y + cards_h - 42, body, card_w - 20, 7.2, 10, MUTED)

    _text(
        c,
        MARGIN,
        50,
        "Code: _tool_height_offsets() / run_trial() | Evidence: approach-clearance-comparison.json and paired.csv",
        6.2,
        MUTED,
        "Courier",
    )
    c.showPage()


def _draw_extended_approach_clearance(c: canvas.Canvas, root: Path) -> None:
    report = json.loads(
        (root / "results" / "cooperative-pick-approach-clearance-extended-comparison.json").read_text()
    )
    conditions = report["outcomes_by_condition"]
    paired = report["paired_outcomes"]
    contacts = report["contact_attribution"]
    metadata = json.loads(
        (root / "assets" / "demos" / "cooperative_pick_approach_clearance_extended_pair.json").read_text()
    )
    baseline_mm = round(report["baseline"]["lowest_approach_tool_height_offset_m"] * 1000)
    candidate_mm = round(report["candidate"]["lowest_approach_tool_height_offset_m"] * 1000)

    _header(c, 25, "Independent approach validation")
    _title(
        c,
        "The gain repeats across 60 new poses",
        "A second seed block improves both camera-guided and oracle diagnostic scores. The paired reversals and overlapping confidence intervals still argue for caution.",
    )

    panel_gap = 12
    panel_width = (CONTENT_W - panel_gap) / 2
    panel_y, panel_h = 466, 204
    for index, (condition, title) in enumerate(
        (("camera_estimate", "Camera estimate"), ("oracle_xy", "Oracle XY diagnostic"))
    ):
        x = MARGIN + index * (panel_width + panel_gap)
        _rounded(c, x, panel_y, panel_width, panel_h, PANEL, stroke=GRID, radius=10)
        _text(c, x + 12, panel_y + panel_h - 22, title, 9, WHITE, "Helvetica-Bold")
        metric_base = conditions[condition]["baseline"]["physical_stable_lift_success"]
        metric_candidate = conditions[condition]["candidate"]["physical_stable_lift_success"]
        pair = paired[condition]["physical_stable_lift_success"]
        for row_index, (name, metric, color) in enumerate(
            ((f"{baseline_mm} mm", metric_base, BLUE), (f"{candidate_mm} mm", metric_candidate, GREEN))
        ):
            row_y = panel_y + panel_h - 57 - row_index * 26
            _text(c, x + 12, row_y + 2, name, 7.2, MUTED)
            bar_x, bar_w = x + 54, panel_width - 105
            c.setFillColor(PANEL_ALT)
            c.roundRect(bar_x, row_y, bar_w, 12, 3, fill=1, stroke=0)
            c.setFillColor(color)
            c.roundRect(bar_x, row_y, max(2, bar_w * metric["rate"]), 12, 3, fill=1, stroke=0)
            _right_text(c, x + panel_width - 11, row_y + 2, f"{metric['successes']}/60", 7.5, WHITE, "Helvetica-Bold")
        base_ci = metric_base["wilson_95_interval"]
        candidate_ci = metric_candidate["wilson_95_interval"]
        _text(
            c,
            x + 12,
            panel_y + 85,
            f"Wilson 95%: {base_ci[0]:.0%}-{base_ci[1]:.0%} vs. {candidate_ci[0]:.0%}-{candidate_ci[1]:.0%}",
            6.3,
            MUTED,
        )
        _text(
            c,
            x + 12,
            panel_y + 65,
            f"Paired wins/losses: {pair['candidate_only_success']} candidate-only / {pair['baseline_only_success']} baseline-only",
            6.6,
            GREEN if pair["candidate_only_success"] >= pair["baseline_only_success"] else ORANGE,
        )
        net = metric_candidate["successes"] - metric_base["successes"]
        _text(
            c,
            x + 12,
            panel_y + 36,
            f"Net change: {net:+d} of 60",
            8.5,
            GREEN if net >= 0 else RED,
            "Helvetica-Bold",
        )

    _section_label(c, MARGIN, 441, "Contact trials over the complete pick cycle", ORANGE)
    contact_y, contact_h = 302, 125
    _rounded(c, MARGIN, contact_y, CONTENT_W, contact_h, PANEL_ALT, stroke=GRID, radius=9)
    for index, (condition, label) in enumerate(
        (("camera_estimate", "Camera estimate"), ("oracle_xy", "Oracle XY diagnostic"))
    ):
        x = MARGIN + index * (CONTENT_W / 2) + 11
        _text(c, x, contact_y + contact_h - 19, label, 7.8, WHITE, "Helvetica-Bold")
        arm_base = contacts["baseline"][condition]["arm_environment"]["trials_with_contact"]
        arm_candidate = contacts["candidate"][condition]["arm_environment"]["trials_with_contact"]
        payload_base = contacts["baseline"][condition]["nonfinger_payload"]["trials_with_contact"]
        payload_candidate = contacts["candidate"][condition]["nonfinger_payload"]["trials_with_contact"]
        _text(c, x, contact_y + 76, f"Arm-table/ground: {arm_base}/60 → {arm_candidate}/60", 7.2, MUTED)
        _text(c, x, contact_y + 54, f"Nonfinger arm-payload: {payload_base}/60 → {payload_candidate}/60", 7.2, MUTED)
        _text(c, x, contact_y + 25, "Trials with any contact; categories can overlap.", 6.1, SUBTLE)

    _section_label(c, MARGIN, 277, "Matched replay and interpretation", GREEN)
    image_path = root / "assets" / "demos" / "cooperative_pick_approach_clearance_extended_contact.png"
    image = ImageReader(str(image_path))
    c.drawImage(image, MARGIN, 140, width=320, height=92, preserveAspectRatio=True, anchor="c", mask="auto")
    _text(c, MARGIN, 127, f"Pose {metadata['pair_id']} | same camera-estimate target, fail → pass", 6.8, MUTED)
    _rounded(c, MARGIN + 331, 83, CONTENT_W - 331, 160, PANEL, stroke=GRID, radius=8)
    _text(c, MARGIN + 343, 225, "Evidence boundary", 8.2, ORANGE, "Helvetica-Bold")
    _paragraph(
        c,
        MARGIN + 343,
        207,
        "This is a 60-pose MuJoCo split, not hardware evidence. The 4 camera baseline-only outcomes remain visible in the paired counts. Camera intrinsics are ideal and fixed; the target is detected once; fresh post-pick visual confirmation is not measured. Keep the 20 mm setting opt-in.",
        CONTENT_W - 355,
        7.1,
        9.3,
        MUTED,
    )
    _text(
        c,
        MARGIN,
        50,
        "Code: compare_cooperative_pick_geometry.py | Evidence: extended-comparison.json, extended-paired.csv, and paired replay metadata",
        5.9,
        MUTED,
        "Courier",
    )
    c.showPage()


def _draw_cooperative_pick_visual_confirmation(c: canvas.Canvas, root: Path) -> None:
    summary = json.loads(
        (
            root
            / "results"
            / "cooperative-pick-visual-confirmation"
            / "cooperative-pick-visual-confirmation-summary.json"
        ).read_text()
    )
    conditions = summary["conditions"]
    paired_physics = summary["paired_physical_comparison"]["outcomes"]
    classifier_names = (
        ("translation_only_baseline", "V0  rise + drift"),
        ("orientation_only_baseline", "V1  + fiducial angle"),
        ("rise_orientation_baseline", "V2  + higher rise gate"),
        ("image_classifier", "V3  tighter angle gate"),
    )
    aggregate: dict[str, dict[str, int]] = {}
    for key, _ in classifier_names:
        aggregate[key] = {
            field: sum(
                condition["results"][key].get(field, 0)
                for condition in conditions.values()
            )
            for field in ("trials", "true_positive", "false_confirmation", "true_negative", "false_nonconfirmation", "abstained", "confirmed", "not_confirmed")
        }
    total_positive = sum(
        condition["results"]["capture_time_ground_truth"]["positive"]
        for condition in conditions.values()
    )
    total_trials = sum(condition["results"]["trials"] for condition in conditions.values())

    _header(c, 26, "Fresh visual verification")
    _title(
        c,
        "A second view checks the cooperative lift",
        "After the initial overhead image plans the motion, a fixed side camera takes two fresh images. The image decision and the simulator's physical audit remain separate.",
    )

    _section_label(c, MARGIN, 678, "Three RGB observations from one simulated pick", GREEN)
    tile_gap = 10
    tile_w = (CONTENT_W - tile_gap * 2) / 3
    tile_y, tile_h = 545, 112
    for index, label in enumerate(("START  |  0.0 s", "FRESH  |  8.4 s", "FRESH  |  8.8 s")):
        x = MARGIN + index * (tile_w + tile_gap)
        _rounded(c, x, tile_y, tile_w, tile_h, PANEL, stroke=GRID, radius=8)
        _text(c, x + 9, tile_y + tile_h - 17, label, 7.4, WHITE, "Helvetica-Bold")
        window_x, window_y = x + 18, tile_y + 25
        c.setFillColor(PANEL_ALT)
        c.roundRect(window_x, window_y, tile_w - 36, 55, 5, fill=1, stroke=0)
        marker_x = x + tile_w / 2
        marker_y = window_y + (14 if index == 0 else 39 if index == 1 else 37)
        c.setFillColor(PURPLE)
        c.roundRect(marker_x - 5, marker_y - 11, 10, 22, 2, fill=1, stroke=0)
        c.setStrokeColor(GREEN)
        c.setLineWidth(1.2)
        c.line(marker_x + 14, window_y + 14, marker_x + 14, window_y + 42)
        c.line(marker_x + 14, window_y + 42, marker_x + 10, window_y + 36)
        c.line(marker_x + 14, window_y + 42, marker_x + 18, window_y + 36)
        _text(c, x + 9, tile_y + 10, "magenta fiducial | image-space measurement", 6.1, MUTED)

    left_x, right_x = MARGIN, MARGIN + CONTENT_W / 2 + 7
    panel_w = CONTENT_W / 2 - 7
    panel_y, panel_h = 308, 214
    _rounded(c, left_x, panel_y, panel_w, panel_h, PANEL, stroke=GRID, radius=9)
    _section_label(c, left_x + 12, panel_y + panel_h - 20, "Classifier changes on the held-out split", BLUE)
    _text(c, left_x + 12, panel_y + panel_h - 40, "Same frames for V0 / V1 / V2 / V3", 7.5, MUTED)
    _table(
        c,
        left_x + 10,
        panel_y + panel_h - 52,
        [100, 42, 42, 42],
        [["Image rule", "False\nconfirm", "Recall", "Coverage"]]
        + [
            [
                label,
                str(aggregate[key]["false_confirmation"]),
                f"{aggregate[key]['true_positive'] / total_positive:.0%}" if total_positive else "n/a",
                f"{(aggregate[key]['confirmed'] + aggregate[key]['not_confirmed']) / total_trials:.0%}" if total_trials else "n/a",
            ]
            for key, label in classifier_names
        ],
        font_size=6.3,
        leading=8,
    )
    _paragraph(
        c,
        left_x + 12,
        panel_y + 29,
        "V0 checks rise and drift; V1 adds angle; V2 raises minimum rise to 50 px; V3 tightens angle to 4 degrees. All four match on seed 39400 (112 TP, 8 TN, zero errors or abstentions), so V3 shows no measured gain here. Eight negative labels leave specificity uncertain.",
        panel_w - 24,
        6.2,
        7.8,
        MUTED,
    )

    _rounded(c, right_x, panel_y, panel_w, panel_h, PANEL, stroke=GRID, radius=9)
    _section_label(c, right_x + 12, panel_y + panel_h - 20, "Physical outcomes - separate score", ORANGE)
    _text(c, right_x + 12, panel_y + panel_h - 40, f"{total_trials} trials across two matched approach profiles", 7.2, MUTED)
    y = panel_y + panel_h - 68
    for condition_name, label in (("baseline_5mm", "5 mm baseline"), ("experimental_20mm", "20 mm experimental")):
        result = conditions[condition_name]["results"]["physical_outcomes"]
        _text(c, right_x + 12, y, label, 7.2, WHITE, "Helvetica-Bold")
        _text(c, right_x + 112, y, f"hold {result['payload_stable_hold_successes']}/{conditions[condition_name]['results']['trials']}", 7.0, GREEN)
        _text(c, right_x + 220, y, f"strict {result['strict_full_sequence_successes']}/{conditions[condition_name]['results']['trials']}", 7.0, BLUE)
        y -= 24
    _paragraph(
        c,
        right_x + 12,
        y - 1,
        f"Same-pose strict comparison: 20 mm had {paired_physics['strict_full_sequence_success']['experimental_20mm_only_successes']} candidate-only wins and {paired_physics['strict_full_sequence_success']['baseline_5mm_only_successes']} baseline-only wins. Classifier labels, full hold, and full-cycle collision safety remain separate.",
        panel_w - 24,
        6.6,
        8.5,
        MUTED,
    )

    _callout(
        c,
        MARGIN,
        70,
        CONTENT_W,
        224,
        "Evidence boundary and next failure",
        "The first translation-only validation produced two false confirmations when the payload was highly tilted and had no grasp contact. Development then exposed an under-lift false confirmation, which motivated the 50 px rise gate. A later split found a 5.23 degree payload tilt that passed the 12 degree rule, so the angle gate was tightened to 4 degrees. Seed 39400 is the independent evaluation of V3. This remains an ideal synthetic camera with a visual-only fiducial and two samples; it does not prove continuous holding, safe release, or real-camera robustness. Keep the 20 mm approach opt-in.",
        ORANGE,
        7.1,
    )
    _text(
        c,
        MARGIN,
        50,
        "Code: cooperative_pick_visual_verification.py / cooperative_pick_visual_validation.py | Data: seed-39400 summary, CSV, and SVG scorecard",
        5.4,
        MUTED,
        "Courier",
    )
    c.showPage()


def _draw_cooperative_pick_visual_faults(c: canvas.Canvas, root: Path) -> None:
    summary = json.loads(
        (
            root
            / "results"
            / "cooperative-pick-visual-fault-validation"
            / "cooperative-pick-visual-fault-summary.json"
        ).read_text()
    )
    conditions = summary["fault_conditions"]
    split = summary["split"]
    _header(c, 27, "IMAGE FAULT VALIDATION")
    _title(
        c,
        "Stress the camera path without changing the pick",
        "V3 thresholds stay frozen while controlled faults alter only the side-camera RGB. The initial overhead control image and MuJoCo physics remain unchanged.",
    )

    _section_label(c, MARGIN, 678, "Keep the image and physical paths separate", GREEN)
    flow_y, flow_h = 598, 62
    gap = 12
    flow_w = (CONTENT_W - gap * 3) / 4
    flow_items = (
        ("CONTROL INPUT", "Initial overhead RGB"),
        ("OBSERVATIONS", "Side RGB at 0.0 / 8.4 / 8.8 s"),
        ("FAULT INJECTION", "Post-capture image transforms"),
        ("EVALUATION", "MuJoCo truth and full motion"),
    )
    for index, (heading, detail) in enumerate(flow_items):
        x = MARGIN + index * (flow_w + gap)
        _rounded(c, x, flow_y, flow_w, flow_h, PANEL, stroke=GRID, radius=7)
        _text(c, x + 8, flow_y + flow_h - 17, heading, 6.4, GREEN if index < 3 else ORANGE, "Helvetica-Bold")
        _paragraph(c, x + 8, flow_y + 27, detail, flow_w - 16, 6.6, 8, MUTED)
        if index < len(flow_items) - 1:
            _arrow(c, x + flow_w + 2, flow_y + flow_h / 2, x + flow_w + gap - 2, flow_y + flow_h / 2, BLUE, 0.9)

    _section_label(
        c,
        MARGIN,
        574,
        "Frozen V3 against capture-time labels | 58 positive / 2 negative per condition",
        BLUE,
    )
    labels = {
        "clean": "Clean RGB",
        "dim_exposure_x0_25": "Dim exposure x0.25",
        "bright_exposure_x1_5": "Bright exposure x1.5",
        "marker_occlusion_50pct": "50% marker occlusion",
        "horizontal_blur_9px": "Horizontal blur 9 px",
        "camera_shift_up_25px": "Fresh camera shift up 25 px",
    }
    table_rows = [["Image condition", "False\nconfirm", "False\nreject", "Abstain", "Coverage", "Recall", "Specificity"]]
    for key in labels:
        result = conditions[key]["results"]
        table_rows.append(
            [
                labels[key],
                str(result["false_confirmation"]),
                str(result["false_nonconfirmation"]),
                str(result["abstained"]),
                f"{result['coverage']:.0%}",
                f"{result['positive_recall_including_abstentions']:.0%}" if result["positive_recall_including_abstentions"] is not None else "n/a",
                f"{result['negative_specificity_including_abstentions']:.0%}" if result["negative_specificity_including_abstentions"] is not None else "n/a",
            ]
        )
    _table(
        c,
        MARGIN + 4,
        552,
        [142, 52, 52, 58, 65, 70, 70],
        table_rows,
        font_size=7.0,
        leading=8,
    )

    dim = conditions["dim_exposure_x0_25"]["results"]
    shifted = conditions["camera_shift_up_25px"]["results"]
    all_physics_same = summary["physical_outcomes_identical_across_faults_for_every_pose"]
    body = (
        f"Seed {split['base_seed']} covers {split['matched_target_poses']} matched poses in "
        f"{split['trial_count']} runs. The seed-39500 smoke check was excluded from validation. "
        f"Dimming produced {dim['abstained']} abstentions and {dim['false_confirmation']} false confirmations; "
        f"a 25 px fresh-frame shift produced {shifted['false_confirmation']} false confirmations. "
        "Only two capture-time negatives were present, so observed specificity is weak evidence. "
        f"Physical truth and full-motion outcomes matched across each pose's six repeats: "
        f"{'yes' if all_physics_same else 'no'}. These are authored image transforms, not real-camera data. "
        "Next, test gap-tolerant marker components and an image-quality abstention gate on development seeds, then keep the change opt-in until a separate frozen split validates it."
    )
    _callout(
        c,
        MARGIN,
        78,
        CONTENT_W,
        178,
        "Evidence boundary and next engineering step",
        body,
        ORANGE,
        7.0,
    )
    _text(
        c,
        MARGIN,
        50,
        "Code: visual_image_faults.py / cooperative_pick_visual_fault_validation.py | Data: seed-39600 summary, CSV, and SVG scorecard",
        5.4,
        MUTED,
        "Courier",
    )
    c.showPage()


def _draw_cooperative_pick_occlusion(c: canvas.Canvas, root: Path) -> None:
    development = json.loads(
        (
            root
            / "results"
            / "cooperative-pick-occlusion-development"
            / "cooperative-pick-occlusion-summary.json"
        ).read_text()
    )
    heldout = json.loads(
        (
            root
            / "results"
            / "cooperative-pick-occlusion-heldout"
            / "cooperative-pick-occlusion-summary.json"
        ).read_text()
    )
    selected = development["selected_gap_tolerance_px"]
    _header(c, 28, "POST-PICK OCCLUSION VALIDATION")
    _title(
        c,
        "Rejoin a split fiducial with bounded evidence",
        "The measured failure was partial occlusion: the marker split into components and V3 rejected real lifts. V4 groups aligned fragments while measuring only pixels that remain visible.",
    )

    card_y, card_h = 545, 132
    left_x, right_x = MARGIN, PAGE_W / 2 + 5
    card_w = (CONTENT_W - 10) / 2
    _rounded(c, left_x, card_y, card_w, card_h, PANEL, stroke=GRID, radius=8)
    _rounded(c, right_x, card_y, card_w, card_h, PANEL, stroke=GRID, radius=8)
    _section_label(c, left_x + 12, card_y + card_h - 18, "Observed components", ORANGE)
    _section_label(c, right_x + 12, card_y + card_h - 18, "V4 bounded union", GREEN)
    for panel_x, tolerant in ((left_x, False), (right_x, True)):
        marker_x = panel_x + 45
        marker_bottom, marker_top = card_y + 33, card_y + 99
        c.setFillColor(PURPLE)
        c.roundRect(marker_x, marker_bottom, 13, 30, 3, fill=1, stroke=0)
        c.roundRect(marker_x, marker_bottom + 36, 13, 30, 3, fill=1, stroke=0)
        c.setFillColor(HexColor("#606b79"))
        c.rect(marker_x - 7, marker_bottom + 30, 27, 6, fill=1, stroke=0)
        c.setStrokeColor(GREEN if tolerant else ORANGE)
        c.setLineWidth(1.1)
        c.setDash(3, 2) if tolerant else c.setDash()
        c.rect(marker_x - 4, marker_bottom - 2, 21, 70, fill=0, stroke=1)
        c.setDash()
        if tolerant:
            c.setStrokeColor(GREEN)
            c.line(marker_x + 27, marker_bottom + 2, marker_x + 27, marker_top - 2)
            c.line(marker_x + 27, marker_top - 2, marker_x + 24, marker_top - 8)
            c.line(marker_x + 27, marker_top - 2, marker_x + 30, marker_top - 8)
            _text(c, marker_x + 41, card_y + 84, f"{selected} px max gap", 7.1, GREEN, "Helvetica-Bold")
            _text(c, marker_x + 41, card_y + 68, "2 visible parts", 7.1, MUTED)
            _text(c, marker_x + 41, card_y + 52, "union from pixels", 7.1, MUTED)
        else:
            _text(c, marker_x + 30, card_y + 84, "Middle band hidden", 7.1, ORANGE, "Helvetica-Bold")
            _text(c, marker_x + 30, card_y + 68, "V3 chooses one", 7.1, MUTED)
            _text(c, marker_x + 30, card_y + 52, "largest component", 7.1, MUTED)
    _text(c, MARGIN, 528, "A gap is not filled: the detector uses the coordinates of observed pixels for its center and PCA angle.", 7.0, MUTED)

    selected_v4 = f"gap_tolerant_{selected}px"
    development_poses = development["split"]["matched_target_poses"]
    heldout_poses = heldout["split"]["matched_target_poses"]
    table_y, table_h = 280, 192
    _rounded(c, left_x, table_y, card_w, table_h, PANEL, stroke=GRID, radius=8)
    _rounded(c, right_x, table_y, card_w, table_h, PANEL, stroke=GRID, radius=8)
    for panel_x, split, title, poses in (
        (left_x, development, "DEVELOPMENT | SEED 39700", development_poses),
        (right_x, heldout, "FROZEN VALIDATION | SEED 39800", heldout_poses),
    ):
        _section_label(c, panel_x + 12, table_y + table_h - 18, title, BLUE)
        _text(c, panel_x + 12, table_y + table_h - 34, f"{poses} matched poses | same V3 thresholds", 6.8, MUTED)
        y = table_y + table_h - 57
        for condition, label in (("clean", "Clean"), ("marker_occlusion_50pct", "50% occlusion")):
            v3 = split["conditions"][condition]["v3"]
            v4 = split["conditions"][condition][selected_v4]
            row_text = (
                f"{label}: V3 TP {v3['true_positive']}/{v3['capture_time_truth_positive']}, "
                f"FP {v3['false_confirmation']}, FN {v3['false_nonconfirmation']}"
            )
            _text(c, panel_x + 12, y, row_text, 6.7, WHITE, "Helvetica-Bold")
            y -= 13
            row_text = (
                f"V4 TP {v4['true_positive']}/{v4['capture_time_truth_positive']}, "
                f"FP {v4['false_confirmation']}, FN {v4['false_nonconfirmation']}"
            )
            _text(c, panel_x + 22, y, row_text, 6.7, GREEN)
            y -= 25
        physical = split["conditions"]["clean"]["physical_outcomes"]
        _text(
            c,
            panel_x + 12,
            table_y + 17,
            f"Physics unchanged by image condition: hold {physical['payload_stable_hold_successes']}/{physical['trials']}; strict {physical['strict_full_sequence_successes']}/{physical['trials']}",
            6.1,
            ORANGE,
        )

    _callout(
        c,
        MARGIN,
        82,
        CONTENT_W,
        126,
        "What the comparison supports",
        f"On the development split, the frozen {selected} px option preserved all clean V3 decisions and raised occluded positive recall from {development['conditions']['marker_occlusion_50pct']['v3']['true_positive']}/{development['conditions']['marker_occlusion_50pct']['v3']['capture_time_truth_positive']} to {development['conditions']['marker_occlusion_50pct'][selected_v4]['true_positive']}/{development['conditions']['marker_occlusion_50pct'][selected_v4]['capture_time_truth_positive']}. The separate seed-39800 split tests this one frozen option. Each split contains few negative labels; these results support better tolerance to this authored marker band only. V3 remains the default, and a visual confirmation remains separate from physical hold and full-cycle collision safety.",
        ORANGE,
        6.7,
    )
    _text(
        c,
        MARGIN,
        50,
        "Code: cooperative_pick_visual_verification.py / cooperative_pick_occlusion_validation.py | Data: seeds 39700 and 39800, CSV and SVG scorecards",
        5.1,
        MUTED,
        "Courier",
    )
    c.showPage()


def _draw_cooperative_pick_occlusion_challenge(c: canvas.Canvas, root: Path) -> None:
    report = json.loads(
        (
            root
            / "results"
            / "cooperative-pick-occlusion-shape-challenge"
            / "cooperative-pick-occlusion-shape-summary.json"
        ).read_text()
    )
    strata = report["strata"]
    nominal = strata["nominal_pad_16mm"]
    hard = strata["small_pad_14mm_hard_negative"]
    _header(c, 29, "FROZEN OCCLUSION CHALLENGE")
    _title(
        c,
        "Gap tolerance helps only some occlusions",
        "The 28 px option was frozen before this challenge. Each physical pose was simulated once, then nine image variants were evaluated from the same initial and two fresh side-camera frames.",
    )

    card_y, card_h = 625, 73
    card_w = (CONTENT_W - 12) / 2
    for x, title, result, color in (
        (MARGIN, "NOMINAL PADS | 100 POSES", nominal, BLUE),
        (MARGIN + card_w + 12, "14 mm HARD-NEGATIVE PADS | 100 POSES", hard, ORANGE),
    ):
        _rounded(c, x, card_y, card_w, card_h, PANEL, stroke=GRID, radius=8)
        _section_label(c, x + 11, card_y + card_h - 17, title, color)
        physics = result["physical_outcomes"]
        _text(
            c,
            x + 11,
            card_y + 36,
            f"Capture-time truth: +{physics['capture_time_truth_positive']} / −{physics['capture_time_truth_negative']}",
            8.1,
            WHITE,
            "Helvetica-Bold",
        )
        _text(
            c,
            x + 11,
            card_y + 18,
            f"Payload hold {physics['payload_stable_hold_successes']}/100 | strict full cycle {physics['strict_full_sequence_successes']}/100",
            7.1,
            MUTED,
        )

    _section_label(c, MARGIN, 594, "Authored side-camera masks | shown relative to the fiducial", PURPLE)
    mask_items = [
        ("horizontal_25_center", "H band 25%"),
        ("horizontal_50_upper", "H upper 50%"),
        ("horizontal_50_lower", "H lower 50%"),
        ("horizontal_75_center", "H band 75%"),
        ("vertical_30_center", "V band 30%"),
        ("diagonal_25_plus45", "Diagonal +45°"),
        ("diagonal_25_minus45", "Diagonal −45°"),
        ("patch_45_upper_left", "Corner patch"),
    ]
    mini_w, mini_h, mini_gap = 118, 48, 10
    mini_top_rows = (581, 526)
    for index, (condition, label) in enumerate(mask_items):
        col, row = index % 4, index // 4
        x = MARGIN + col * (mini_w + mini_gap)
        y = mini_top_rows[row] - mini_h
        _rounded(c, x, y, mini_w, mini_h, PANEL, stroke=GRID, radius=5)
        _text(c, x + 6, y + mini_h - 12, label, 6.5, WHITE, "Helvetica-Bold")
        marker_x = x + mini_w / 2 - 5
        marker_y = y + 5
        marker_h = 27
        c.setFillColor(PURPLE)
        c.roundRect(marker_x, marker_y, 10, marker_h, 2, fill=1, stroke=0)
        c.setFillColor(HexColor("#606b79"))
        if condition.startswith("horizontal"):
            band_h = {"horizontal_25_center": 4, "horizontal_50_upper": 8, "horizontal_50_lower": 8, "horizontal_75_center": 12}[condition]
            center = marker_y + marker_h / 2
            if condition.endswith("upper"):
                center += 4
            elif condition.endswith("lower"):
                center -= 4
            c.rect(marker_x - 3, center - band_h / 2, 16, band_h, fill=1, stroke=0)
        elif condition == "vertical_30_center":
            c.rect(marker_x + 3.5, marker_y - 2, 3, marker_h + 4, fill=1, stroke=0)
        elif condition.startswith("diagonal"):
            direction = 1 if condition.endswith("plus45") else -1
            c.setStrokeColor(HexColor("#606b79"))
            c.setLineWidth(4)
            c.line(
                marker_x + (0 if direction > 0 else 10),
                marker_y + 4,
                marker_x + (10 if direction > 0 else 0),
                marker_y + marker_h - 4,
            )
        else:
            c.setFillColor(HexColor("#606b79"))
            c.rect(marker_x, marker_y + marker_h * 0.55, 5, marker_h * 0.45, fill=1, stroke=0)

    table_rows = [
        ["Image condition", "16 mm positives\nV3 → V4 / 94", "14 mm positives\nV3 → V4 / 18", "False confirms\nV3 → V4 / 82 negatives"],
    ]
    for condition, label in (
        ("clean", "Clean"),
        ("horizontal_25_center", "Horizontal 25% centered"),
        ("horizontal_50_upper", "Horizontal 50% upper"),
        ("horizontal_50_lower", "Horizontal 50% lower"),
        ("horizontal_75_center", "Horizontal 75% centered"),
        ("vertical_30_center", "Vertical 30% centered"),
        ("diagonal_25_plus45", "Diagonal +45°"),
        ("diagonal_25_minus45", "Diagonal −45°"),
        ("patch_45_upper_left", "Upper-left patch"),
    ):
        nominal_item = nominal["image_conditions"][condition]
        hard_item = hard["image_conditions"][condition]
        n_v3, n_v4 = nominal_item["v3"], nominal_item["v4_frozen_28px"]
        h_v3, h_v4 = hard_item["v3"], hard_item["v4_frozen_28px"]
        table_rows.append(
            [
                label,
                f"{n_v3['true_positive']} → {n_v4['true_positive']}",
                f"{h_v3['true_positive']} → {h_v4['true_positive']}",
                f"{h_v3['false_confirmation']} → {h_v4['false_confirmation']}",
            ]
        )
    _table(
        c,
        MARGIN,
        465,
        [177, 111, 111, CONTENT_W - 399],
        table_rows,
        font_size=6.6,
        leading=7.5,
    )

    _callout(
        c,
        MARGIN,
        49,
        CONTENT_W,
        130,
        "What the challenge changed",
        "Moderate horizontal bands let V4 recover some true lifts, but the vertical, diagonal, and patch masks show no gain. At 75% band height, nominal true positives fell from 37/94 to 6/94. In the 14 mm hard-negative stratum, the 25% mask added one false confirmation and the severe mask still produced five among 82 capture-time negatives. This mask/pad mix is an authored stress test, not a deployment prevalence estimate. Keep V4 opt-in and V3 as the default. Capture-time visual truth remains separate from payload hold and full-cycle safety.",
        ORANGE,
        6.8,
    )
    _text(
        c,
        MARGIN,
        38,
        "Code: visual_image_faults.py / cooperative_pick_occlusion_challenge.py | Data: seeds 40100 and 40200, 1,800 image rows",
        5.0,
        MUTED,
        "Courier",
    )
    c.showPage()


def _draw_orthogonal_verification(c: canvas.Canvas, root: Path) -> None:
    report = json.loads(
        (
            root
            / "results"
            / "cooperative-pick-orthogonal-verification-heldout"
            / "cooperative-pick-orthogonal-verification-summary.json"
        ).read_text()
    )
    summaries = report["results"]
    methods = ("V3_side", "V4_side_28px", "V4_two_view")
    labels = ("V3 | side", "V4 | side 28px", "V4 + orthogonal")
    totals = {}
    for method, label in zip(methods, labels, strict=True):
        totals[label] = {
            key: sum(summaries[name]["methods"][method][key] for name in summaries)
            for key in (
                "true_positive",
                "false_confirmation",
                "true_negative",
                "false_nonconfirmation",
                "abstained",
            )
        }
    total_positives = sum(item["capture_truth_positive"] for item in summaries.values())
    total_negatives = sum(item["capture_truth_negative"] for item in summaries.values())
    total_runs = report["total_unique_physical_pose_runs"]
    _header(c, 30, "ORTHOGONAL FRESH VERIFICATION")
    _title(
        c,
        "A second view catches a hidden tilt",
        "The latest held-out experiment renders an actual MuJoCo rail across one camera's sight line, then checks the same free payload from two perpendicular side views.",
    )

    image_path = root / "assets" / "demos" / "cooperative_pick_orthogonal_verification_card.png"
    image = ImageReader(str(image_path))
    image_w = CONTENT_W
    image_h = image_w * 624 / 1280
    image_y = 445
    _rounded(c, MARGIN - 2, image_y - 2, image_w + 4, image_h + 4, PANEL, stroke=GRID, radius=7)
    c.drawImage(image, MARGIN, image_y, width=image_w, height=image_h, mask="auto")
    _text(
        c,
        MARGIN,
        image_y - 14,
        "Counted held-out replay: original camera rail, independent view, image verdict, and separate simulator audit",
        6.7,
        MUTED,
    )

    card_y, card_h = 369, 42
    card_w = (CONTENT_W - 16) / 3
    cards = (
        ("240", "physical pose runs", BLUE),
        ("121", "capture-time negatives", ORANGE),
        ("0 / 121", "two-view false confirms", GREEN),
    )
    for index, (value, label, accent) in enumerate(cards):
        x = MARGIN + index * (card_w + 8)
        _rounded(c, x, card_y, card_w, card_h, PANEL, stroke=GRID, radius=7)
        _text(c, x + 9, card_y + 24, value, 12, accent, "Helvetica-Bold")
        _text(c, x + 68, card_y + 24, label, 7.1, MUTED)

    table_rows = [["Image rule", "TP / 119", "False confirms / 121", "TN / 121", "Abstain", "Coverage"]]
    for label in labels:
        result = totals[label]
        coverage = total_runs - result["abstained"]
        table_rows.append(
            [
                label,
                str(result["true_positive"]),
                str(result["false_confirmation"]),
                str(result["true_negative"]),
                str(result["abstained"]),
                f"{coverage}/{total_runs}",
            ]
        )
    _table(
        c,
        MARGIN,
        351,
        [126, 62, 105, 63, 62, CONTENT_W - 418],
        table_rows,
        font_size=7.1,
        leading=8.0,
    )

    nominal = summaries["nominal_pad_16mm"]
    hard = summaries["small_pad_14mm_negative_control"]
    stratum_card_y, stratum_card_h = 190, 56
    stratum_card_w = (CONTENT_W - 12) / 2
    for index, (title, result, color) in enumerate(
        (
            ("16 mm nominal pads", nominal, BLUE),
            ("14 mm hard-negative control", hard, ORANGE),
        )
    ):
        x = MARGIN + index * (stratum_card_w + 12)
        _rounded(c, x, stratum_card_y, stratum_card_w, stratum_card_h, PANEL, stroke=GRID, radius=7)
        _section_label(c, x + 9, stratum_card_y + stratum_card_h - 15, title, color)
        _text(
            c,
            x + 9,
            stratum_card_y + 25,
            f"Capture truth +{result['capture_truth_positive']} / −{result['capture_truth_negative']}",
            7.3,
            WHITE,
            "Helvetica-Bold",
        )
        candidate_row = result["methods"]["V4_two_view"]
        _text(
            c,
            x + 9,
            stratum_card_y + 10,
            f"Two-view TP {candidate_row['true_positive']} | false confirms {candidate_row['false_confirmation']} | abstain {candidate_row['abstained']}",
            6.7,
            MUTED,
        )

    physical = [
        summaries[name]
        for name in ("nominal_pad_16mm", "small_pad_14mm_negative_control")
    ]
    hold = sum(item["stable_hold_successes"] for item in physical)
    strict = sum(item["strict_full_cycle_successes"] for item in physical)
    candidate = totals["V4 + orthogonal"]
    callout_y, callout_h = 53, 122
    _callout(
        c,
        MARGIN,
        callout_y,
        CONTENT_W,
        callout_h,
        "A safer decision, with more abstentions",
        f"The two-view option confirms {candidate['true_positive']}/{total_positives} positives and makes 0/{total_negatives} false confirmations (Wilson 95% upper bound 3.1%). It abstains on {candidate['abstained']} poses and misses 19 positives; side-only V4 misses 18. Across the same 240 trajectories, {hold}/240 pass the full payload-hold check and {strict}/240 pass the strict full cycle. These physical outcomes are independent of the image decision. Keep V3 as the runtime default; V4 and two-view V4 remain opt-in.",
        ORANGE,
        7.2,
    )
    _text(
        c,
        MARGIN,
        40,
        "Code: cooperative_pick_visual_verification.py / cooperative_pick_orthogonal_validation.py | Data: seeds 40500 and 40600, 240 poses",
        5.1,
        MUTED,
        "Courier",
    )
    c.showPage()


def _draw_cooperative_pick_material(c: canvas.Canvas, root: Path) -> None:
    report = json.loads(
        (
            root
            / "results"
            / "cooperative-pick-material-validation"
            / "cooperative-pick-material-summary.json"
        ).read_text()
    )
    conditions = report["conditions"]
    _header(c, 31, "MASS AND FRICTION STRESS")
    _title(
        c,
        "Contact has a friction threshold",
        "A paired MuJoCo stress map separates the camera target estimate from the shared grasp and contact trajectory.",
    )

    card_y, card_h = 661, 43
    card_w = (CONTENT_W - 16) / 3
    cards = (
        ("216", "paired physics trials", BLUE),
        ("12", "shared target poses", GREEN),
        ("9", "mass/friction settings", ORANGE),
    )
    for index, (value, label, accent) in enumerate(cards):
        x = MARGIN + index * (card_w + 8)
        _rounded(c, x, card_y, card_w, card_h, PANEL, stroke=GRID, radius=7)
        _text(c, x + 10, card_y + 24, value, 13, accent, "Helvetica-Bold")
        _text(c, x + 50, card_y + 24, label, 7.2, MUTED)
        _text(c, x + 10, card_y + 9, "base seed 40800 | same poses in every cell", 6.4, SUBTLE)

    masses = tuple(report["varied_material_assumptions"]["payload_body_mass_kg"])
    frictions = tuple(report["varied_material_assumptions"]["finger_pad_and_payload_sliding_friction"])
    panels = (
        ("camera_estimate", "Camera-estimated target", MARGIN),
        ("oracle_xy", "Oracle target upper bound", PAGE_W / 2 + 4),
    )
    panel_w = CONTENT_W / 2 - 6
    top = 609
    for controller, label, panel_x in panels:
        _rounded(c, panel_x, 350, panel_w, 253, PANEL, stroke=GRID, radius=8)
        _section_label(c, panel_x + 10, top - 17, label, BLUE)
        _text(c, panel_x + 10, top - 39, "pad/payload sliding friction", 6.6, MUTED)
        grid_x = panel_x + 82
        cell_w, cell_h, col_gap = 48, 49, 5
        for col, friction in enumerate(frictions):
            x = grid_x + col * (cell_w + col_gap)
            _text(c, x + 10, top - 39, f"{friction:g}", 7, GREEN, "Helvetica-Bold")
        for row_index, mass in enumerate(masses):
            y = top - 105 - row_index * 59
            mass_label = "0.72 kg*" if abs(mass - 0.7201166534540158) < 1e-8 else f"{mass:.2f} kg"
            _text(c, panel_x + 9, y + 25, mass_label, 6.6, WHITE, "Helvetica-Bold")
            for col, friction in enumerate(frictions):
                key = f"mass-{mass:g}_friction-{friction:g}"
                record = conditions[key]["controllers"][controller]
                strict = record["physical_stable_lift_success"]["successes"]
                hold = record["payload_stable_lift_success"]["successes"]
                cycle = record["full_sequence_collision_safe"]["successes"]
                rate = strict / record["trials"]
                fill = Color(
                    0.12 + 0.10 * rate,
                    0.16 + 0.48 * rate,
                    0.25 + 0.34 * rate,
                )
                x = grid_x + col * (cell_w + col_gap)
                _rounded(c, x, y, cell_w, cell_h, fill, stroke=GRID, radius=5)
                _text(c, x + 6, y + 28, f"{strict}/12", 9.1, WHITE, "Helvetica-Bold")
                _text(c, x + 5, y + 12, f"H {hold} / C {cycle}", 5.7, MUTED)
        _text(c, panel_x + 10, 365, "Strict / 12     H = stable hold     C = full-cycle collision-safe", 5.5, MUTED)

    _text(c, MARGIN + 9, 337, "* Nominal mass: 0.720116653454 kg; original compiled inertia preserved.", 7, MUTED)
    _callout(
        c,
        MARGIN,
        132,
        CONTENT_W,
        110,
        "What the result says",
        "At friction 0.5, neither controller achieved a strict success in any mass cell; all 72 runs were labeled no grasp contact. At friction 2.2, camera-estimated success was 11/12, 10/12, and 11/12 across light, nominal, and heavy payloads. Matching oracle failures at low friction point to contact mechanics or the shared trajectory, while the small, author-selected simulator sweep cannot establish real-object reliability.",
        ORANGE,
        7.7,
    )
    _text(
        c,
        MARGIN,
        111,
        "Visual confirmation: not measured | Code: cooperative_pick_material_validation.py | Data: 216 paired rows",
        6.3,
        MUTED,
        "Courier",
    )
    _text(
        c,
        MARGIN,
        96,
        "The CSV preserves strict lift, hold-only, and full-cycle collision outcomes as separate columns.",
        6.3,
        MUTED,
    )
    c.showPage()


def _draw_cooperative_pick_pad_friction(c: canvas.Canvas, root: Path) -> None:
    report = json.loads(
        (
            root
            / "results"
            / "cooperative-pick-pad-friction-validation"
            / "cooperative-pick-pad-friction-summary.json"
        ).read_text()
    )
    conditions = report["conditions"]
    _header(c, 32, "Held-out pad geometry")
    _title(
        c,
        "A larger pad did not restore hold contact",
        "New-seed paired MuJoCo test: vary pad half-size while keeping nominal payload, friction, pose, approach, release, and controller matched.",
    )

    card_y, card_h = 661, 43
    card_w = (CONTENT_W - 16) / 3
    cards = (
        ("96", "paired physics trials", BLUE),
        ("12", "shared target poses", GREEN),
        ("2 × 2", "pad and friction settings", ORANGE),
    )
    for index, (value, label, accent) in enumerate(cards):
        x = MARGIN + index * (card_w + 8)
        _rounded(c, x, card_y, card_w, card_h, PANEL, stroke=GRID, radius=7)
        _text(c, x + 10, card_y + 24, value, 13, accent, "Helvetica-Bold")
        _text(c, x + 54, card_y + 24, label, 7.2, MUTED)
        _text(c, x + 10, card_y + 9, "seed 40900 | camera estimate + oracle diagnostic", 6.2, SUBTLE)

    pad_sizes = tuple(report["varied_settings"]["finger_pad_half_size_m"])
    for friction, panel_x in ((0.5, MARGIN), (2.2, PAGE_W / 2 + 4)):
        panel_w = CONTENT_W / 2 - 6
        panel_y, panel_h = 363, 267
        _rounded(c, panel_x, panel_y, panel_w, panel_h, PANEL, stroke=GRID, radius=8)
        _section_label(c, panel_x + 10, panel_y + panel_h - 20, f"Sliding friction μ = {friction:g}", BLUE)
        for controller, label, row_top in (
            ("camera_estimate", "Camera estimate", panel_y + 207),
            ("oracle_xy", "Oracle XY diagnostic", panel_y + 111),
        ):
            _text(c, panel_x + 10, row_top + 13, label, 7.2, WHITE, "Helvetica-Bold")
            controller_rows = conditions[f"friction-{friction:g}"]["controllers"][controller]
            for index, pad_size in enumerate(pad_sizes):
                record = controller_rows[f"pad_half_size_m_{pad_size:g}"]
                strict = record["physical_stable_lift_success"]
                hold = record["payload_stable_lift_success"]["successes"]
                cycle = record["full_sequence_collision_safe"]["successes"]
                row_y = row_top - 12 - index * 38
                _text(c, panel_x + 10, row_y + 5, f"{pad_size * 1000:.0f} mm", 6.6, MUTED)
                bar_x, bar_w = panel_x + 67, 105
                c.setFillColor(PANEL_ALT)
                c.roundRect(bar_x, row_y, bar_w, 15, 4, fill=1, stroke=0)
                rate_w = bar_w * strict["successes"] / report["target_sampling"]["samples_per_seed"]
                if rate_w > 0:
                    c.setFillColor(SUBTLE if index == 0 else GREEN)
                    c.roundRect(bar_x, row_y, rate_w, 15, 4, fill=1, stroke=0)
                _text(c, bar_x + bar_w + 7, row_y + 4, f"{strict['successes']}/12", 6.8, WHITE, "Helvetica-Bold")
                _text(c, panel_x + 67, row_y - 9, f"hold {hold}   |   full-cycle safe {cycle}", 5.8, MUTED)

    _callout(
        c,
        MARGIN,
        229,
        CONTENT_W,
        100,
        "Measured conclusion",
        "At friction 0.5, both pad sizes produced 0/12 strict lifts for camera and oracle; the legacy no_grasp_contact label means zero contacts in the scored hold window. At friction 2.2, both pad sizes scored 12/12 in both controllers. The 18 mm candidate did not improve low-friction hold contact, so the runtime 16 mm geometry stays in place.",
        ORANGE,
        7.5,
    )
    _text(c, MARGIN + 3, 202, "NEXT: phase traces find contact in closure/early lift, then none in loaded hold; measure relative slip next.", 7.4, BLUE, "Helvetica-Bold")
    _text(c, MARGIN, 91, "Hold and full-cycle collision safety remain separate metrics; visual confirmation: not measured.", 6.5, MUTED)
    _text(c, MARGIN, 76, "Code: cooperative_pick_pad_friction_validation.py | Data: seed 40900, 96 paired rows", 6.3, MUTED, "Courier")
    _text(c, MARGIN, 61, "The four-panel video replays one counted pose and adds no benchmark trials.", 6.3, MUTED)
    c.showPage()


def _draw_cooperative_pick_contact_diagnostics(c: canvas.Canvas, root: Path) -> None:
    report = json.loads(
        (
            root
            / "results"
            / "cooperative-pick-contact-diagnostics"
            / "cooperative-pick-contact-summary.json"
        ).read_text()
    )
    conditions = report["conditions"]
    _header(c, 33, "Contact retention diagnostic")
    _title(
        c,
        "Contact disappears before hold",
        "A new 48-trial block samples six named finger pads at each MuJoCo physics step to separate closure, cooperative lift, and loaded hold.",
    )

    card_y, card_h = 661, 43
    card_w = (CONTENT_W - 16) / 3
    cards = (
        ("48", "paired physics trials", BLUE),
        ("288", "per-finger traces", GREEN),
        ("2 ms", "physics timestep", ORANGE),
    )
    for index, (value, label, accent) in enumerate(cards):
        x = MARGIN + index * (card_w + 8)
        _rounded(c, x, card_y, card_w, card_h, PANEL, stroke=GRID, radius=7)
        _text(c, x + 10, card_y + 24, value, 13, accent, "Helvetica-Bold")
        _text(c, x + 55, card_y + 24, label, 7.2, MUTED)
        _text(c, x + 10, card_y + 9, "seed 41000 | same poses at mu 0.5 and 2.2", 6.2, SUBTLE)

    pad_trials = report["target_sampling"]["samples_per_seed"] * 6
    phase_rows = (
        ("Approach / close / secure", "pad_trials_with_contact_during_approach_close_or_secure", BLUE),
        ("Cooperative lift", "pad_trials_with_contact_during_cooperative_lift", ORANGE),
        ("Hold under load", "pad_trials_with_contact_during_hold_under_load", GREEN),
    )
    for friction, panel_x in ((0.5, MARGIN), (2.2, PAGE_W / 2 + 4)):
        panel_w = CONTENT_W / 2 - 6
        panel_y, panel_h = 340, 295
        _rounded(c, panel_x, panel_y, panel_w, panel_h, PANEL, stroke=GRID, radius=8)
        _section_label(c, panel_x + 10, panel_y + panel_h - 20, f"Sliding friction μ = {friction:g}", BLUE)
        for controller, label, title_y in (
            ("camera_estimate", "Camera estimate", panel_y + 250),
            ("oracle_xy", "Oracle XY diagnostic", panel_y + 122),
        ):
            summary = conditions[f"friction-{friction:g}"][controller]
            _text(c, panel_x + 10, title_y, label, 7.3, WHITE, "Helvetica-Bold")
            for index, (phase_label, field, color) in enumerate(phase_rows):
                count = summary[field]
                bar_y = title_y - 20 - index * 29
                bar_x, bar_w = panel_x + 132, 96
                _text(c, panel_x + 10, bar_y + 4, phase_label, 5.8, MUTED)
                c.setFillColor(PANEL_ALT)
                c.roundRect(bar_x, bar_y, bar_w, 14, 4, fill=1, stroke=0)
                if count:
                    c.setFillColor(color)
                    c.roundRect(bar_x, bar_y, bar_w * count / pad_trials, 14, 4, fill=1, stroke=0)
                _text(c, bar_x + bar_w + 5, bar_y + 3, f"{count}/{pad_trials}", 6.2, WHITE, "Helvetica-Bold")
            strict = summary["physical_stable_lift_success"]["successes"]
            _text(c, panel_x + 10, title_y - 111, f"Strict lift {strict}/12   |   hold {summary['payload_stable_lift_successes']}   |   cycle-safe {summary['full_sequence_collision_safe_trials']}", 6.3, ORANGE)

    _callout(
        c,
        MARGIN,
        211,
        CONTENT_W,
        105,
        "What the phase data resolves",
        "At mu 0.5, both controllers show pad contact during closure and early lift, but none of the 24 trials retains any pad contact in the loaded hold. At mu 2.2, camera-estimate succeeds 12/12 and oracle succeeds 11/12. The low-friction failure is contact retention through lift; this study does not distinguish tangential slip from finger separation.",
        ORANGE,
        7.4,
    )
    _text(c, MARGIN + 3, 183, "The old no_grasp_contact label means zero contacts in its scored hold window; it does not mean the fingers never touched.", 6.9, BLUE, "Helvetica-Bold")
    _text(c, MARGIN, 93, "Contact wrench is a MuJoCo solver output, not a calibrated sensor. Visual confirmation: not measured.", 6.3, MUTED)
    _text(c, MARGIN, 78, "Code: cooperative_pick_contact_diagnostics.py | Data: seed 41000, 48 paired rows", 6.3, MUTED, "Courier")
    _text(c, MARGIN, 63, "Next: test one opt-in grip-preload change on new held-out poses.", 6.3, MUTED)
    c.showPage()


def _draw_cooperative_pick_retention_diagnostics(c: canvas.Canvas, root: Path) -> None:
    report_path = (
        root / "results" / "cooperative-pick-retention-diagnostics"
        / "cooperative-pick-retention-summary.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    conditions = report["conditions"]
    sample_count = report["target_sampling"]["samples_per_seed"]
    _header(c, 34, "Low-friction failure diagnosis")
    _title(
        c,
        "The payload slips relative to closed fingers",
        "A fresh 48-trial split measures hand-relative payload motion, pad contact forces, and finger position tracking through the loaded hold.",
    )

    card_y, card_h = 665, 46
    card_w = (CONTENT_W - 16) / 3
    for index, (value, label, accent) in enumerate((
        ("48", "paired physics trials", BLUE),
        ("576", "pad and phase records", GREEN),
        ("20 ms", "diagnostic sample period", ORANGE),
    )):
        x = MARGIN + index * (card_w + 8)
        _rounded(c, x, card_y, card_w, card_h, PANEL, stroke=GRID, radius=7)
        _text(c, x + 10, card_y + 25, value, 13, accent, "Helvetica-Bold")
        _text(c, x + 70, card_y + 25, label, 7.4, MUTED)
        _text(c, x + 10, card_y + 9, "seed 42000 | 12 matched poses per condition", 6.2, SUBTLE)

    _rounded(c, MARGIN, 398, CONTENT_W, 232, PANEL, stroke=GRID, radius=9)
    _section_label(c, MARGIN + 14, 611, "Median maximum downward payload drift relative to the gripper hand (mm)", BLUE)
    _rounded(c, PAGE_W - MARGIN - 168, 585, 10, 10, ORANGE, radius=2)
    _text(c, PAGE_W - MARGIN - 153, 587, "mu 0.5", 7.2, MUTED)
    _rounded(c, PAGE_W - MARGIN - 99, 585, 10, 10, GREEN, radius=2)
    _text(c, PAGE_W - MARGIN - 84, 587, "mu 2.2", 7.2, MUTED)

    chart_x, chart_top, chart_width = MARGIN + 187, 565, 282
    scale_m = 0.17
    for tick_mm in (0, 40, 80, 120, 160):
        tick_x = chart_x + chart_width * tick_mm / (scale_m * 1000)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.45)
        c.line(tick_x, 421, tick_x, chart_top + 2)
        _text(c, tick_x - 6, 407, str(tick_mm), 6.2, MUTED)
    for controller, controller_label, y_base in (
        ("camera_estimate", "Camera estimate", 527),
        ("oracle_xy", "Oracle XY", 456),
    ):
        _text(c, MARGIN + 14, y_base + 8, controller_label, 8, WHITE, "Helvetica-Bold")
        for friction, color, row_y in ((0.5, ORANGE, y_base + 14), (2.2, GREEN, y_base - 9)):
            summary = conditions[f"friction-{friction:g}"][controller]
            drift_mm = 1000 * summary[
                "maximum_downward_payload_drift_in_hand_frame_m_median"
            ]
            c.setFillColor(PANEL_ALT)
            c.roundRect(chart_x, row_y, chart_width, 12, 4, fill=1, stroke=0)
            c.setFillColor(color)
            c.roundRect(
                chart_x, row_y,
                chart_width * min(drift_mm / (scale_m * 1000), 1.0), 12,
                4, fill=1, stroke=0,
            )
            _text(c, chart_x + chart_width + 8, row_y + 2, f"{drift_mm:.1f}", 7.3, WHITE, "Helvetica-Bold")

    low_camera = conditions["friction-0.5"]["camera_estimate"]
    low_oracle = conditions["friction-0.5"]["oracle_xy"]
    high_camera = conditions["friction-2.2"]["camera_estimate"]
    high_oracle = conditions["friction-2.2"]["oracle_xy"]
    low_camera_safe = low_camera["collision_safe_trials"]
    low_oracle_safe = low_oracle["collision_safe_trials"]
    high_camera_safe = high_camera["collision_safe_trials"]
    high_oracle_safe = high_oracle["collision_safe_trials"]
    camera_ratio = low_camera[
        "mean_tangential_to_normal_force_ratio_when_contacting_median"
    ]
    oracle_ratio = low_oracle[
        "mean_tangential_to_normal_force_ratio_when_contacting_median"
    ]
    camera_drift_mm = 1000 * high_camera[
        "maximum_downward_payload_drift_in_hand_frame_m_median"
    ]
    oracle_drift_mm = 1000 * high_oracle[
        "maximum_downward_payload_drift_in_hand_frame_m_median"
    ]
    _callout(
        c,
        MARGIN,
        265,
        CONTENT_W,
        112,
        "Evidence points to payload slip, not an open gripper",
        f"At mu 0.5, all {sample_count * 2} camera/oracle trials lost pad contact during hold. Median maximum finger-slide error during hold stayed at {1000 * low_camera['maximum_slide_tracking_error_during_hold_m_median']:.3f} mm (camera) and {1000 * low_oracle['maximum_slide_tracking_error_during_hold_m_median']:.3f} mm (oracle). Contacting-sample tangential/normal force ratios were {camera_ratio:.3f} and {oracle_ratio:.3f}, near the selected friction value. At mu 2.2, both controllers lifted {high_camera['physical_stable_lift_successes']}/{sample_count}, with {camera_drift_mm:.0f} and {oracle_drift_mm:.0f} mm median hand-relative drift.",
        ORANGE,
        8.0,
    )
    _callout(
        c,
        MARGIN,
        113,
        CONTENT_W,
        130,
        "Safety and confirmation remain separate",
        f"Only {low_camera_safe}/{sample_count} camera and {low_oracle_safe}/{sample_count} oracle low-friction runs passed the full-cycle collision checks; at high friction {high_camera_safe}/{sample_count} camera and {high_oracle_safe}/{sample_count} oracle runs passed. These are simulation outcomes under one scripted grasp and author-selected friction. No fresh visual confirmation was measured. Next, test a single opt-in grip-preload candidate on new matched poses and keep physical lift, collision safety, and image confirmation as separate scores.",
        BLUE,
        8.2,
    )
    _text(c, MARGIN, 82, "Code: cooperative_pick_retention_diagnostics.py | Data: seed 42000, 48 trials, 576 pad-phase rows", 6.3, MUTED, "Courier")
    c.showPage()


def main() -> Path:
    repository = Path(__file__).resolve().parents[1]
    output = repository / "output" / "pdf" / "visual-servo-mujoco-project-explainer.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
    c.setTitle("Visual Servo MuJoCo Project Explainer")
    c.setAuthor("Jad")
    c.setSubject("Visual servoing project architecture, code, benchmarks, and fresh pick verification")
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
    _draw_multiarm_integration(c, repository)
    _draw_release_profile(c, repository)
    _draw_approach_clearance(c, repository)
    _draw_extended_approach_clearance(c, repository)
    _draw_cooperative_pick_visual_confirmation(c, repository)
    _draw_cooperative_pick_visual_faults(c, repository)
    _draw_cooperative_pick_occlusion(c, repository)
    _draw_cooperative_pick_occlusion_challenge(c, repository)
    _draw_orthogonal_verification(c, repository)
    _draw_cooperative_pick_material(c, repository)
    _draw_cooperative_pick_pad_friction(c, repository)
    _draw_cooperative_pick_contact_diagnostics(c, repository)
    _draw_cooperative_pick_retention_diagnostics(c, repository)
    c.save()
    print(f"Wrote {output}")
    return output


if __name__ == "__main__":
    main()
