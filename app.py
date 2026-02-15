import cv2
import streamlit as st
import time
import pygame
import pandas as pd
from src.posture import detect_posture
from src.fatigue import FatigueTracker

st.title("Real-Time Posture & Fatigue Monitor")

run = st.checkbox("Start Webcam")

frame_window = st.image([])
dashboard = st.empty()   #single dashboard container

tracker = FatigueTracker()
pygame.mixer.init()
beep_sound = pygame.mixer.Sound("assets/beep.wav")
beep_channel = pygame.mixer.Channel(1)
beep_channel.stop()
break_sound = pygame.mixer.Sound("assets/break.wav")
break_channel = pygame.mixer.Channel(2)
break_channel.stop()

last_posture_beep = 0
cooldown = 5

break_active = False
break_start_time = 0
break_duration = 8
last_break_trigger = 0
break_cooldown_until = 0
absent_start = None
absence_reset_time = 5

# ---------- DASHBOARD DATA ----------
good_frames = 0
bad_frames = 0
fatigue_history = []
time_history = []
alert_count = 0

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera error")
        break

    frame = cv2.flip(frame, 1)

    posture, forward = detect_posture(frame)
    tracker.update(posture)

    bad_time = tracker.get_bad_duration()
    fatigue_score = tracker.get_fatigue_score()

    # -------- TRACK DATA --------
    if posture == "Good":
        good_frames += 1
    elif posture == "Bad":
        bad_frames += 1

    fatigue_history.append(fatigue_score)
    time_history.append(time.time() - tracker.session_start)

    # -------- TEXT --------
    cv2.putText(frame, f"Posture: {posture}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame, f"Slouch level: {forward}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.putText(frame, f"Bad posture time: {bad_time}s", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(frame, f"Fatigue score: {fatigue_score}", (20,160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    current_time = time.time()

    # ---------- ABSENCE ----------
    if posture == "Unknown":
        if absent_start is None:
            absent_start = current_time

        last_posture_beep = current_time

        if current_time - absent_start > absence_reset_time:
            tracker = FatigueTracker()
            last_break_trigger = current_time

            if break_active:
                break_channel.stop()
                break_active = False
    else:
        absent_start = None

    # ---------- POSTURE BEEP ----------
    if bad_time > 6:
        cv2.putText(frame, "Sit Straight!", (20,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if current_time - last_posture_beep > cooldown:
            beep_channel.play(beep_sound)
            last_posture_beep = current_time
            alert_count += 1

    # ---------- BREAK ALERT ----------
    if (
        fatigue_score > 20
        and not break_active
        and posture != "Unknown"
        and current_time > break_cooldown_until
    ):
        if current_time - last_break_trigger > cooldown:
            break_active = True
            break_start_time = current_time
            last_break_trigger = current_time
            break_channel.play(break_sound, loops=-1)
            alert_count += 1

    # ---------- BREAK CONTROL ----------
    if break_active:
        if current_time - break_start_time >= break_duration:
            break_channel.stop()
            break_active = False
            break_cooldown_until = current_time + 20

        elif posture == "Unknown":
            break_channel.stop()
            break_active = False
            break_cooldown_until = current_time + 20

        else:
            cv2.putText(frame, "Take a Break!", (20,250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    frame_window.image(frame, channels="BGR")

    # ================= DASHBOARD =================
    with dashboard.container():

        st.subheader("Session Dashboard")

        total_frames = good_frames + bad_frames
        good_pct = (good_frames/total_frames*100) if total_frames else 0
        bad_pct = (bad_frames/total_frames*100) if total_frames else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Good Posture %", f"{good_pct:.1f}%")
        col2.metric("Bad Posture %", f"{bad_pct:.1f}%")
        col3.metric("Alerts Triggered", alert_count)

        session_duration = int(time.time() - tracker.session_start)
        st.write(f"Session Duration: {session_duration}s")
        st.write(f"Total Bad Posture Time: {int(tracker.total_bad_time)}s")

        risk_score = min(100, fatigue_score + bad_pct)
        st.metric("Risk Score", int(risk_score))

        if len(fatigue_history) > 10:
            df = pd.DataFrame({
                "Time": time_history,
                "Fatigue": fatigue_history
            })
            st.line_chart(df.set_index("Time"))

cap.release()
