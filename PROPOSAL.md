# Trixel

**Tech for Good 2026** · GDG Coimbatore · Build weekend Aug 8–9, GRD College

**Track:** AI for Good Health & Well-being
**Team code:** TEAM-264

## Problem

Bhuvaneshwari is recovering from breast surgery. She has to perform one physiotherapy exercise (Elbow Winging) at home every day. She isn't sure whether she's doing it correctly, and doing it incorrectly can slow recovery or reinforce poor movement patterns.

## Who it helps

It is aimed at helping Bhuvaneshwari, a 50 year old recovering from a cyst removal surgery in the breast area. Due to the surgery she was advised to do arm physio. The tool will help her family members and physical therapist monitor her exercise progress. It will also give feedback and insights on the accuracy and effectiveness of her movements.

## Solution

We propose a lightweight web application that helps Bhuvaneshwari, a patient recovering from breast surgery, correctly perform her prescribed Elbow Winging physiotherapy exercise at home. The physiotherapist records one reference demonstration of the exercise. Using only a phone or laptop webcam, the application captures Bhuvaneshwari performing the same movement.

MediaPipe extracts shoulder, elbow and wrist landmarks from both videos. The application compares elbow angles and movement trajectories to determine whether the exercise matches the therapist's demonstration.

Therapist login-> uploads correct exercise videos + defines key movement -> stored in reference exercise database
               
Patient logs in -> performs exercise (recorded on web camera) -> AI Pose Estimation & Motion Analysis (Extracts body landmarks and movement)

After completing the exercise, the patient receives spoken and visual feedback indicating whether the movement matched the therapist's reference and what should be corrected before the next repetition.

Final goal: Bhuvaneshwari can independently perform her prescribed Elbow Winging exercise at home with immediate AI feedback that closely matches her physiotherapist's demonstration.

## Architecture

Webcam video → MediaPipe hand-landmark detection → vector-based pattern comparison (live vs. stored reference) → threshold-based correctness decision → feedback + progress logging.

The core modules are:
1. Hand Tracking
2. Training (record reference)
3. Rehabilitation (compare + feedback)
4. Data storage layer for saved movements 
5. Web portal (Patient Portal / Staff Portal) for access and monitoring.

## Tech stack

React, Node.js, Express, MongoDB, Python, MediaPipe, OpenCV, JWT Auth

## Getting started

1. Accept your collaborator invite (check your email / GitHub notifications).
2. Clone this repo and start building.
3. Commit early and often — this repo is what you present on the day.

---

_Created automatically when your proposal was validated._