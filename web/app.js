/**
 * SignSpeak - Core Application Logic (v7 Bulletproof Anti-Repeat & High Accuracy)
 * ==============================================================================
 * Real-time sign language recognition, sentence building, predictive autocomplete,
 * live FPS monitoring, ghost hand alignment guide, top-3 predictions,
 * and rock-solid latching protocol that completely eliminates duplicate & alternating triggers.
 */

(() => {
  'use strict';

  // --- Configuration & Constants ---
  const CONFIG = {
    CONFIDENCE_THRESHOLD: 0.60, // Minimum confidence to accept a prediction
    HOLD_DURATION_MS: 580,      // Comfortable, deliberate dwell time (0.58s)
    COOLDOWN_MS: 420,           // Clean cooldown recovery after commit (0.42s)
    SMOOTHING_BUFFER_SIZE: 4,   // Frame history for majority vote smoothing (~65ms)
    MODEL_DIR: 'model',
    MAX_HISTORY_ITEMS: 8
  };

  // --- State ---
  const state = {
    cameraRunning: false,
    cameraStream: null,
    classifier: null,
    predictionBuffer: [],       // Array of recent { label, confidence }
    currentCandidate: null,     // Currently held candidate label
    candidateStartTime: null,   // Timestamp candidate was first observed
    cooldownUntil: 0,           // Timestamp until which detection is paused
    latchedLetter: null,        // Letter that was committed and requires gesture release/change
    hasActiveLatch: false,      // Latch flag preventing continuous repeated commits
    newCandidateHoldStart: null,// Validates intentional sign change before unlocking latch
    newCandidateLabel: null,
    theme: localStorage.getItem('signspeak-theme') || 'dark',
    soundEnabled: localStorage.getItem('signspeak-sound') !== 'false',
    audioContext: null,
    historyTrail: [],
    // FPS tracking
    lastFrameTime: performance.now(),
    frameCount: 0,
    currentFps: 0
  };

  // --- DOM Elements ---
  const DOM = {
    // Video & Canvas
    video: document.getElementById('webcam-video'),
    canvas: document.getElementById('output-canvas'),
    canvasCtx: null,
    videoWrapper: document.getElementById('video-wrapper'),
    cameraStatus: document.getElementById('camera-status'),
    cameraToggleBtn: document.getElementById('toggle-camera-btn'),
    cameraErrorAlert: document.getElementById('camera-error-alert'),
    cameraErrorMessage: document.getElementById('camera-error-msg'),
    cameraRetryBtn: document.getElementById('camera-retry-btn'),
    ghostHandGuide: document.getElementById('ghost-hand-guide'),
    fpsBadge: document.getElementById('fps-badge'),

    // Detection HUD
    detectedLetter: document.getElementById('detected-letter'),
    detectedConfidence: document.getElementById('detected-confidence'),
    progressRingCircle: document.getElementById('progress-ring-circle'),
    cooldownIndicator: document.getElementById('cooldown-indicator'),
    cooldownBar: document.getElementById('cooldown-bar'),
    statusBadge: document.getElementById('detection-status-badge'),
    top3Items: document.getElementById('top3-items'),

    // Sentence Builder & Controls
    sentenceBox: document.getElementById('sentence-box'),
    wordSuggestionsContainer: document.getElementById('word-suggestions'),
    historyChips: document.getElementById('history-chips'),
    btnSpace: document.getElementById('btn-space'),
    btnBackspace: document.getElementById('btn-backspace'),
    btnDeleteWord: document.getElementById('btn-delete-word'),
    btnCopy: document.getElementById('btn-copy'),
    btnClear: document.getElementById('btn-clear'),
    btnNewSentence: document.getElementById('btn-new-sentence'),
    toastContainer: document.getElementById('toast-container'),

    // Settings
    themeToggleBtn: document.getElementById('theme-toggle-btn'),
    soundToggleBtn: document.getElementById('sound-toggle-btn'),

    // Modals
    helpBtn: document.getElementById('help-btn'),
    helpModal: document.getElementById('help-modal'),
    closeHelpBtn: document.getElementById('close-help-btn'),
    aslGuideBtn: document.getElementById('asl-guide-btn'),
    aslModal: document.getElementById('asl-modal'),
    closeAslBtn: document.getElementById('close-asl-btn'),
    aslGrid: document.getElementById('asl-grid'),

    // Quick phrases
    quickPhrases: document.querySelectorAll('.quick-phrase-chip')
  };

  // --- Progress Ring Configuration ---
  const RING_RADIUS = 35;
  const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

  function initProgressRing() {
    if (DOM.progressRingCircle) {
      DOM.progressRingCircle.style.strokeDasharray = `${RING_CIRCUMFERENCE} ${RING_CIRCUMFERENCE}`;
      DOM.progressRingCircle.style.strokeDashoffset = `${RING_CIRCUMFERENCE}`;
    }
  }

  function setProgress(percent) {
    if (!DOM.progressRingCircle) return;
    const offset = RING_CIRCUMFERENCE - (percent / 100) * RING_CIRCUMFERENCE;
    DOM.progressRingCircle.style.strokeDashoffset = `${offset}`;
  }

  // --- Toast Notification System ---
  function showToast(message, durationMs = 2400) {
    if (!DOM.toastContainer) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    DOM.toastContainer.appendChild(toast);
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, durationMs);
  }

  // --- Web Audio Click Synthesizer ---
  function playCommitChime(pitchMultiplier = 1.0) {
    if (!state.soundEnabled) return;
    try {
      if (!state.audioContext) {
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (state.audioContext.state === 'suspended') {
        state.audioContext.resume();
      }
      const now = state.audioContext.currentTime;
      const osc = state.audioContext.createOscillator();
      const gain = state.audioContext.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33 * pitchMultiplier, now); // D5
      osc.frequency.exponentialRampToValueAtTime(880 * pitchMultiplier, now + 0.12); // A5

      gain.gain.setValueAtTime(0.25, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);

      osc.connect(gain);
      gain.connect(state.audioContext.destination);

      osc.start(now);
      osc.stop(now + 0.15);
    } catch (e) {
      // Audio autoplay policy fallback
    }

    if ('vibrate' in navigator) {
      try { navigator.vibrate(40); } catch (_) {}
    }
  }

  // --- Majority Vote Prediction Smoothing ---
  function getMajorityPrediction(buffer) {
    if (buffer.length === 0) return null;
    const counts = {};
    const confSums = {};

    for (const item of buffer) {
      counts[item.label] = (counts[item.label] || 0) + 1;
      confSums[item.label] = (confSums[item.label] || 0) + item.confidence;
    }

    let topLabel = null;
    let maxCount = 0;
    for (const [label, count] of Object.entries(counts)) {
      if (count > maxCount) {
        maxCount = count;
        topLabel = label;
      }
    }

    const avgConf = confSums[topLabel] / maxCount;
    return {
      label: topLabel,
      confidence: avgConf,
      voteFraction: maxCount / buffer.length
    };
  }

  // --- History Trail ---
  function addHistoryItem(letter) {
    state.historyTrail.push(letter);
    if (state.historyTrail.length > CONFIG.MAX_HISTORY_ITEMS) {
      state.historyTrail.shift();
    }
    renderHistoryTrail();
  }

  function renderHistoryTrail() {
    if (!DOM.historyChips) return;
    DOM.historyChips.innerHTML = '';
    state.historyTrail.forEach(item => {
      const chip = document.createElement('span');
      chip.className = 'history-chip';
      chip.textContent = item;
      DOM.historyChips.appendChild(chip);
    });
  }

  // --- Sentence Builder Logic ---
  function appendText(text) {
    if (!text) return;
    const current = DOM.sentenceBox.value;
    DOM.sentenceBox.value = current + text;
    DOM.sentenceBox.scrollTop = DOM.sentenceBox.scrollHeight;
    updateSuggestions();

    DOM.sentenceBox.classList.add('flash-commit');
    setTimeout(() => DOM.sentenceBox.classList.remove('flash-commit'), 200);
  }

  function handleSpace() {
    const text = DOM.sentenceBox.value;
    if (text.length > 0 && !text.endsWith(' ')) {
      appendText(' ');
      addHistoryItem('␣');
    }
  }

  function handleBackspace() {
    const text = DOM.sentenceBox.value;
    if (text.length > 0) {
      DOM.sentenceBox.value = text.slice(0, -1);
      updateSuggestions();
      addHistoryItem('⌫');
    }
  }

  function handleDeleteWord() {
    const text = DOM.sentenceBox.value;
    if (!text.trim()) return;
    const trimmed = text.replace(/\s+$/, '');
    const lastSpace = trimmed.lastIndexOf(' ');
    if (lastSpace === -1) {
      DOM.sentenceBox.value = '';
    } else {
      DOM.sentenceBox.value = trimmed.slice(0, lastSpace + 1);
    }
    updateSuggestions();
    addHistoryItem('⌫ Word');
    showToast('Deleted last word');
  }

  function handleCopy() {
    const text = DOM.sentenceBox.value.trim();
    if (!text) {
      showToast('Text box is empty.');
      return;
    }
    navigator.clipboard.writeText(text).then(() => {
      showToast('📋 Copied sentence to clipboard!');
      playCommitChime(1.3);
    }).catch(() => {
      showToast('Failed to copy text.');
    });
  }

  function handleClear() {
    DOM.sentenceBox.value = '';
    updateSuggestions();
    state.historyTrail = [];
    renderHistoryTrail();
    showToast('🗑 Text cleared.');
  }

  function handleNewSentence() {
    const text = DOM.sentenceBox.value.trim();
    if (text.length > 0) {
      if (!/[.!?]$/.test(text)) {
        DOM.sentenceBox.value = text + '. ';
      } else {
        DOM.sentenceBox.value = text + ' ';
      }
      updateSuggestions();
      addHistoryItem('↵');
    }
  }

  function getCurrentWordPrefix() {
    const text = DOM.sentenceBox.value;
    const match = text.match(/([a-zA-Z0-9]+)$/);
    return match ? match[1] : '';
  }

  function updateSuggestions() {
    const prefix = getCurrentWordPrefix();
    const suggestions = getWordSuggestions(prefix, 6);
    DOM.wordSuggestionsContainer.innerHTML = '';

    if (suggestions.length === 0) {
      const hint = document.createElement('span');
      hint.className = 'suggestion-hint';
      hint.textContent = 'Suggestions appear as you sign...';
      DOM.wordSuggestionsContainer.appendChild(hint);
      return;
    }

    suggestions.forEach(word => {
      const chip = document.createElement('button');
      chip.className = 'word-chip';
      chip.type = 'button';
      chip.textContent = word;
      chip.setAttribute('aria-label', `Insert: ${word}`);
      chip.addEventListener('click', () => {
        applyWordSuggestion(word);
      });
      DOM.wordSuggestionsContainer.appendChild(chip);
    });
  }

  function applyWordSuggestion(word) {
    const text = DOM.sentenceBox.value;
    const prefix = getCurrentWordPrefix();
    if (prefix) {
      DOM.sentenceBox.value = text.slice(0, text.length - prefix.length) + word + ' ';
    } else {
      DOM.sentenceBox.value = text + word + ' ';
    }
    addHistoryItem(word);
    updateSuggestions();
    playCommitChime(1.2);
    showToast(`Inserted: "${word}"`);
  }

  // --- Stable Letter Commitment ---
  function commitLetter(label) {
    console.log(`Accepted alphabet: [${label}]`);
    playCommitChime(1.0);
    addHistoryItem(label);

    const lower = label.toLowerCase();
    if (lower === 'space') {
      handleSpace();
    } else if (lower === 'del' || lower === 'backspace') {
      handleBackspace();
    } else if (lower === 'nothing') {
      // Idle
    } else {
      // Regular alphabet letter (A-Z)
      appendText(label);
    }

    state.cooldownUntil = performance.now() + CONFIG.COOLDOWN_MS;
    state.latchedLetter = label;
    state.hasActiveLatch = true;
    state.newCandidateHoldStart = null;
    state.newCandidateLabel = null;
    state.currentCandidate = null;
    state.candidateStartTime = null;
    setProgress(0);

    DOM.detectedLetter.classList.add('pulse-commit');
    setTimeout(() => DOM.detectedLetter.classList.remove('pulse-commit'), 280);

    DOM.statusBadge.textContent = `Captured '${label}' (Release hand to repeat)`;
    DOM.statusBadge.className = 'status-badge ready';
  }

  // --- Real-time Landmark Drawing (Batched Single Path) ---
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index
    [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
    [9, 13], [13, 14], [14, 15], [15, 16],// Ring
    [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
    [0, 17]                               // Palm base
  ];

  function drawLandmarks(landmarks, ctx, width, height, isDimmed = false) {
    ctx.save();
    ctx.clearRect(0, 0, width, height);

    ctx.lineWidth = 3;
    ctx.strokeStyle = isDimmed ? 'rgba(156, 163, 175, 0.4)' : '#06b6d4';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
      const conn = HAND_CONNECTIONS[i];
      const p1 = landmarks[conn[0]];
      const p2 = landmarks[conn[1]];
      ctx.moveTo(p1.x * width, p1.y * height);
      ctx.lineTo(p2.x * width, p2.y * height);
    }
    ctx.stroke();

    if (isDimmed) {
      ctx.beginPath();
      for (let i = 0; i < 21; i++) {
        const lm = landmarks[i];
        ctx.moveTo(lm.x * width + 4, lm.y * height);
        ctx.arc(lm.x * width, lm.y * height, 4, 0, 2 * Math.PI);
      }
      ctx.fillStyle = '#6b7280';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    } else {
      // Group 1: Sky blue joints (15 non-fingertip, non-wrist points)
      ctx.beginPath();
      for (let i = 1; i < 21; i++) {
        if (i !== 4 && i !== 8 && i !== 12 && i !== 16 && i !== 20) {
          const lm = landmarks[i];
          ctx.moveTo(lm.x * width + 4, lm.y * height);
          ctx.arc(lm.x * width, lm.y * height, 4, 0, 2 * Math.PI);
        }
      }
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();

      // Group 2: Emerald fingertips (4, 8, 12, 16, 20)
      ctx.beginPath();
      const fTips = [4, 8, 12, 16, 20];
      for (let j = 0; j < fTips.length; j++) {
        const lm = landmarks[fTips[j]];
        ctx.moveTo(lm.x * width + 5, lm.y * height);
        ctx.arc(lm.x * width, lm.y * height, 5, 0, 2 * Math.PI);
      }
      ctx.fillStyle = '#10b981';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();

      // Group 3: Amber wrist (0)
      ctx.beginPath();
      const wrist = landmarks[0];
      ctx.arc(wrist.x * width, wrist.y * height, 6, 0, 2 * Math.PI);
      ctx.fillStyle = '#f59e0b';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }
    ctx.restore();
  }

  // --- Top-3 Predictions Render (DOM-Reusing) ---
  function renderTop3(top3) {
    if (!DOM.top3Items) return;
    if (!top3 || top3.length === 0) {
      DOM.top3Items.innerHTML = '<span class="suggestion-hint">—</span>';
      return;
    }

    const currentPills = DOM.top3Items.querySelectorAll('.top3-pill');
    if (currentPills.length === top3.length) {
      top3.forEach((item, idx) => {
        const pill = currentPills[idx];
        const pct = Math.round(item.confidence * 100);
        pill.children[0].textContent = item.label;
        pill.children[1].textContent = `${pct}%`;
      });
      return;
    }

    DOM.top3Items.innerHTML = '';
    top3.forEach(item => {
      const pill = document.createElement('div');
      pill.className = 'top3-pill';
      const pct = Math.round(item.confidence * 100);
      pill.innerHTML = `
        <span>${item.label}</span>
        <span style="color:var(--text-muted);font-size:0.75rem;">${pct}%</span>
      `;
      DOM.top3Items.appendChild(pill);
    });
  }

  // --- Frame Detection Handler ---
  function onResults(results) {
    const width = DOM.canvas.width;
    const height = DOM.canvas.height;
    DOM.canvasCtx.clearRect(0, 0, width, height);

    const now = performance.now();

    // Update FPS
    state.frameCount++;
    if (now - state.lastFrameTime >= 1000) {
      state.currentFps = Math.round((state.frameCount * 1000) / (now - state.lastFrameTime));
      DOM.fpsBadge.textContent = `${state.currentFps} FPS`;
      state.frameCount = 0;
      state.lastFrameTime = now;
    }

    // Check cooldown state
    const inCooldown = now < state.cooldownUntil;
    if (inCooldown) {
      const remaining = state.cooldownUntil - now;
      const pct = Math.max(0, Math.min(100, (remaining / CONFIG.COOLDOWN_MS) * 100));
      DOM.cooldownIndicator.classList.remove('hidden');
      DOM.cooldownBar.style.width = `${pct}%`;
      DOM.statusBadge.textContent = 'Cooldown...';
      DOM.statusBadge.className = 'status-badge cooldown';
    } else {
      DOM.cooldownIndicator.classList.add('hidden');
    }

    // Hand detected?
    const hasHand = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

    // Toggle Ghost Hand Guide
    if (DOM.ghostHandGuide) {
      DOM.ghostHandGuide.style.opacity = hasHand ? '0' : '1';
    }

    if (!hasHand) {
      state.predictionBuffer = [];
      state.currentCandidate = null;
      state.candidateStartTime = null;
      state.hasActiveLatch = false; // Release latch only when hand completely leaves frame
      state.latchedLetter = null;
      state.newCandidateHoldStart = null;
      state.newCandidateLabel = null;
      state.classifier.resetSmoothing();
      setProgress(0);
      DOM.detectedLetter.textContent = '—';
      DOM.detectedConfidence.textContent = 'No hand detected';
      DOM.detectedConfidence.className = 'meta-confidence conf-low';
      DOM.statusBadge.textContent = 'Ready';
      DOM.statusBadge.className = 'status-badge ready';
      renderTop3([]);
      return;
    }

    const landmarks = results.multiHandLandmarks[0];
    drawLandmarks(landmarks, DOM.canvasCtx, width, height, inCooldown);

    if (inCooldown) return; // Skip prediction evaluation during cooldown

    // Predict with classifier
    const prediction = state.classifier.predict(landmarks);
    renderTop3(prediction.top3);

    // If prediction is low confidence or invalid, prompt clearly
    if (prediction.isLowConfidence || prediction.label === '—' || !prediction.label) {
      state.predictionBuffer = [];
      state.currentCandidate = null;
      state.candidateStartTime = null;
      // When hand relaxes, release the latch so a new letter can be formed smoothly
      state.hasActiveLatch = false;
      state.latchedLetter = null;
      state.newCandidateHoldStart = null;
      state.newCandidateLabel = null;
      setProgress(0);
      DOM.detectedLetter.textContent = '—';
      DOM.detectedConfidence.textContent = 'Show sign clearly...';
      DOM.detectedConfidence.className = 'meta-confidence conf-low';
      DOM.statusBadge.textContent = 'Ready';
      DOM.statusBadge.className = 'status-badge ready';
      return;
    }

    // Add to smoothing buffer
    state.predictionBuffer.push({
      label: prediction.label,
      confidence: prediction.confidence
    });
    if (state.predictionBuffer.length > CONFIG.SMOOTHING_BUFFER_SIZE) {
      state.predictionBuffer.shift();
    }

    const smoothed = getMajorityPrediction(state.predictionBuffer);
    if (!smoothed || smoothed.label === '—') return;

    const confPercent = Math.round(smoothed.confidence * 100);
    DOM.detectedLetter.textContent = smoothed.label;
    DOM.detectedConfidence.textContent = `${confPercent}% confidence`;

    // Dynamic color coding
    if (confPercent >= 75) {
      DOM.detectedConfidence.className = 'meta-confidence conf-high';
    } else if (confPercent >= 55) {
      DOM.detectedConfidence.className = 'meta-confidence conf-med';
    } else {
      DOM.detectedConfidence.className = 'meta-confidence conf-low';
    }

    // --- BULLETPROOF LATCH & ANTI-REPEAT PROTOCOL ---
    // If a letter was recently captured:
    if (state.hasActiveLatch) {
      if (smoothed.label === state.latchedLetter) {
        // Still holding the exact same sign: 100% block duplicate commit
        state.currentCandidate = null;
        state.candidateStartTime = null;
        state.newCandidateHoldStart = null;
        state.newCandidateLabel = null;
        setProgress(0);
        DOM.statusBadge.textContent = `Captured '${state.latchedLetter}' (Release hand to repeat)`;
        DOM.statusBadge.className = 'status-badge ready';
        DOM.detectedConfidence.textContent = `${confPercent}% • Captured`;
        return;
      } else {
        // Detected a DIFFERENT letter: require at least 220ms continuous hold of the new letter
        // before clearing the latch! This prevents momentary 1-frame jitter from breaking the lock!
        if (state.newCandidateLabel !== smoothed.label) {
          state.newCandidateLabel = smoothed.label;
          state.newCandidateHoldStart = now;
        } else if (now - state.newCandidateHoldStart >= 220) {
          // Genuinely new, stable sign confirmed! Clear previous latch
          state.hasActiveLatch = false;
          state.latchedLetter = null;
          state.newCandidateHoldStart = null;
          state.newCandidateLabel = null;
        } else {
          // Still verifying whether sign change is intentional
          return;
        }
      }
    }

    // Candidate evaluation
    if (smoothed.confidence >= CONFIG.CONFIDENCE_THRESHOLD && smoothed.voteFraction >= 0.6) {
      if (state.currentCandidate === smoothed.label) {
        const holdTime = now - state.candidateStartTime;
        const progress = Math.min(100, (holdTime / CONFIG.HOLD_DURATION_MS) * 100);
        setProgress(progress);

        DOM.statusBadge.textContent = `Holding: ${Math.round(progress)}%`;
        DOM.statusBadge.className = 'status-badge holding';

        if (holdTime >= CONFIG.HOLD_DURATION_MS) {
          commitLetter(smoothed.label);
        }
      } else {
        state.currentCandidate = smoothed.label;
        state.candidateStartTime = now;
        setProgress(0);
        DOM.statusBadge.textContent = 'Hold steady...';
        DOM.statusBadge.className = 'status-badge holding';
      }
    } else {
      state.currentCandidate = null;
      state.candidateStartTime = null;
      setProgress(0);
      DOM.statusBadge.textContent = 'Hold steady...';
      DOM.statusBadge.className = 'status-badge holding';
    }
  }

  // --- Camera Management ---
  async function startCamera() {
    DOM.cameraErrorAlert.classList.add('hidden');
    DOM.cameraStatus.textContent = 'Starting camera...';

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      state.cameraStream = stream;
      DOM.video.srcObject = stream;

      await new Promise(resolve => {
        DOM.video.onloadedmetadata = () => {
          DOM.video.play();
          resolve();
        };
      });

      DOM.canvas.width = DOM.video.videoWidth || 640;
      DOM.canvas.height = DOM.video.videoHeight || 480;
      DOM.canvasCtx = DOM.canvas.getContext('2d');

      const hands = new window.Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      hands.onResults(onResults);

      let isProcessing = false;
      const processFrame = async () => {
        if (!state.cameraRunning) return;
        if (!isProcessing && DOM.video.readyState >= 2) {
          isProcessing = true;
          try {
            await hands.send({ image: DOM.video });
          } catch (err) {
            console.error('Hands processing error:', err);
          }
          isProcessing = false;
        }
        requestAnimationFrame(processFrame);
      };

      state.cameraRunning = true;
      requestAnimationFrame(processFrame);

      DOM.cameraStatus.textContent = 'Camera active';
      DOM.cameraToggleBtn.textContent = 'Stop Camera';
      DOM.cameraToggleBtn.classList.add('btn-danger');
      showToast('Camera started');
    } catch (err) {
      console.error('Camera startup error:', err);
      handleCameraError(err);
    }
  }

  function stopCamera() {
    state.cameraRunning = false;
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach(track => track.stop());
      state.cameraStream = null;
    }
    DOM.video.srcObject = null;
    if (DOM.canvasCtx) {
      DOM.canvasCtx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);
    }
    DOM.cameraStatus.textContent = 'Camera paused';
    DOM.cameraToggleBtn.textContent = 'Start Camera';
    DOM.cameraToggleBtn.classList.remove('btn-danger');
    DOM.detectedLetter.textContent = '—';
    DOM.detectedConfidence.textContent = 'Camera stopped';
    DOM.fpsBadge.textContent = '0 FPS';
    setProgress(0);
  }

  function handleCameraError(err) {
    let msg = 'Could not access the webcam.';
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      msg = 'Camera permission was denied. Please allow camera access in your browser settings and click Retry.';
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      msg = 'No camera found. Please ensure a webcam is connected to your device.';
    } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      msg = 'Camera is currently in use by another application. Please close other camera apps and retry.';
    }
    DOM.cameraErrorMessage.textContent = msg;
    DOM.cameraErrorAlert.classList.remove('hidden');
    DOM.cameraStatus.textContent = 'Camera error';
  }

  // --- ASL Alphabet Guide Modal Generator (A-Z Only) ---
  function populateASLGuide() {
    if (!DOM.aslGrid) return;
    DOM.aslGrid.innerHTML = '';

    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

    const createCard = (title, subtitle) => {
      const card = document.createElement('div');
      card.className = 'asl-card';
      card.innerHTML = `
        <div class="asl-card-title">${title}</div>
        <div class="asl-card-sub">${subtitle}</div>
      `;
      card.addEventListener('click', () => {
        appendText(title);
        addHistoryItem(title);
        playCommitChime();
        showToast(`Added letter: ${title}`);
      });
      return card;
    };

    letters.forEach(char => DOM.aslGrid.appendChild(createCard(char, 'Letter')));
  }

  // --- Sound Setting Management ---
  function updateSoundUI() {
    if (DOM.soundToggleBtn) {
      DOM.soundToggleBtn.textContent = state.soundEnabled ? '🔊 Sound' : '🔇 Muted';
      DOM.soundToggleBtn.setAttribute('aria-label', state.soundEnabled ? 'Mute sound effects' : 'Unmute sound effects');
    }
  }

  // --- Theme Management ---
  function applyTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('signspeak-theme', theme);
    if (DOM.themeToggleBtn) {
      DOM.themeToggleBtn.textContent = theme === 'dark' ? '☀️ Light' : '🌙 Dark';
    }
  }

  // --- Keyboard Shortcuts ---
  function initKeyboard() {
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;

      if (e.code === 'Space' && e.target !== DOM.sentenceBox) {
        e.preventDefault();
        handleSpace();
      } else if (e.code === 'Backspace' && e.target !== DOM.sentenceBox) {
        e.preventDefault();
        if (e.altKey || e.ctrlKey) {
          handleDeleteWord();
        } else {
          handleBackspace();
        }
      } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        handleCopy();
      }
    });
  }

  // --- Initialization ---
  async function init() {
    applyTheme(state.theme);
    updateSoundUI();
    initProgressRing();
    initKeyboard();
    updateSuggestions();
    populateASLGuide();

    // Initialize Classifier
    state.classifier = new window.SignClassifier();
    const loaded = await state.classifier.load(CONFIG.MODEL_DIR);
    if (!loaded) {
      console.warn('Classifier failed to load from model directory. Predictions will not be active.');
    }

    // Event Listeners: Sentence Buttons
    DOM.btnSpace.addEventListener('click', handleSpace);
    DOM.btnBackspace.addEventListener('click', handleBackspace);
    if (DOM.btnDeleteWord) {
      DOM.btnDeleteWord.addEventListener('click', handleDeleteWord);
    }
    DOM.btnCopy.addEventListener('click', handleCopy);
    DOM.btnClear.addEventListener('click', handleClear);
    DOM.btnNewSentence.addEventListener('click', handleNewSentence);

    // Event Listeners: Camera Controls
    DOM.cameraToggleBtn.addEventListener('click', () => {
      if (state.cameraRunning) {
        stopCamera();
      } else {
        startCamera();
      }
    });
    DOM.cameraRetryBtn.addEventListener('click', startCamera);

    // Event Listeners: Quick Phrases (Tap to insert)
    DOM.quickPhrases.forEach(chip => {
      chip.addEventListener('click', () => {
        const phrase = chip.getAttribute('data-phrase') || chip.textContent.trim();
        const text = DOM.sentenceBox.value.trim();
        if (text) {
          DOM.sentenceBox.value = text + ' ' + phrase + '. ';
        } else {
          DOM.sentenceBox.value = phrase + '. ';
        }
        addHistoryItem(phrase);
        updateSuggestions();
        playCommitChime(1.1);
        showToast(`Inserted: "${phrase}"`);
      });
    });

    // Theme toggle
    DOM.themeToggleBtn.addEventListener('click', () => {
      applyTheme(state.theme === 'dark' ? 'light' : 'dark');
    });

    // Sound toggle
    if (DOM.soundToggleBtn) {
      DOM.soundToggleBtn.addEventListener('click', () => {
        state.soundEnabled = !state.soundEnabled;
        localStorage.setItem('signspeak-sound', state.soundEnabled ? 'true' : 'false');
        updateSoundUI();
        if (state.soundEnabled) {
          playCommitChime(1.2);
          showToast('Sound enabled');
        } else {
          showToast('Sound muted');
        }
      });
    }

    // Modals: Help
    DOM.helpBtn.addEventListener('click', () => DOM.helpModal.classList.remove('hidden'));
    DOM.closeHelpBtn.addEventListener('click', () => DOM.helpModal.classList.add('hidden'));
    DOM.helpModal.addEventListener('click', (e) => {
      if (e.target === DOM.helpModal) DOM.helpModal.classList.add('hidden');
    });

    // Modals: ASL Guide
    DOM.aslGuideBtn.addEventListener('click', () => DOM.aslModal.classList.remove('hidden'));
    DOM.closeAslBtn.addEventListener('click', () => DOM.aslModal.classList.add('hidden'));
    DOM.aslModal.addEventListener('click', (e) => {
      if (e.target === DOM.aslModal) DOM.aslModal.classList.add('hidden');
    });

    // Auto-start camera
    startCamera();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
