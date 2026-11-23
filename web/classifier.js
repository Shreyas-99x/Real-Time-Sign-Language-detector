/**
 * SignSpeak - Classifier & Landmark Normalizer (v8 Enhanced Recognition)
 * =========================================================================
 * Hybrid Sign Language Engine combining:
 * 1. Canonical Upright Orientation Normalization (wrist to middle MCP aligned vertically).
 * 2. 4-Layer MLP Neural Network trained on pure ASL single-handed poses (99.97% accuracy).
 * 3. Deep Deterministic Geometric ASL Disambiguation Engine — M/N/T/C/O fist rules.
 * 4. Zero-GC Memory Preallocation for continuous 60 FPS performance without GC stutter.
 * 5. Faster EMA (α=0.65) for snappier response; safe fallbacks for ambiguous poses.
 */

class SignClassifier {
  constructor() {
    this.labels = [];
    this.model = null;          // MLP weights & biases
    this.knnCentroids = null;   // Fallback KNN centroids
    this.isLoaded = false;
    this.emaProbabilities = null;
    this.emaAlpha = 0.65;       // Faster response: 65% weight to new frame, snappier without jitter
    this.lastPredictedLabel = null;

    // Zero-GC Preallocated typed arrays & reusable objects
    this.normVector = new Float32Array(63);
    this.rotLandmarks = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
    this.layerBuffers = [];
    this.rawProbs = null;
    this.finalProbs = null;
    this.indexedProbs = [];
  }

  /**
   * Pre-allocates fixed buffers once classes and model are known.
   */
  initBuffers() {
    const numClasses = this.labels.length || 26;
    this.rawProbs = new Float32Array(numClasses);
    this.finalProbs = new Float32Array(numClasses);
    this.emaProbabilities = new Float32Array(numClasses);
    this.indexedProbs = Array.from({ length: numClasses }, (_, i) => ({
      index: i,
      prob: 0,
      label: this.labels[i] || ''
    }));

    if (this.model && this.model.layer_sizes) {
      this.layerBuffers = this.model.layer_sizes.map(sz => new Float32Array(sz));
    }
  }

  /**
   * Loads model.json, labels.json, and landmarks.json with cache-busting.
   */
  async load(modelDir = 'model') {
    try {
      const cacheBust = `?v=${Date.now()}`;
      console.log(`Loading SignSpeak v8 pure ASL model from '${modelDir}'...`);

      // 1. Load MLP Neural Network
      try {
        const modelRes = await fetch(`${modelDir}/model.json${cacheBust}`);
        if (modelRes.ok) {
          this.model = await modelRes.json();
          this.labels = this.model.classes || [];
          const layersDesc = [this.model.input_dim, ...this.model.layer_sizes].join(' → ');
          console.log(`✓ Loaded pure ASL MLP model: Architecture [${layersDesc}], Accuracy = ${(this.model.accuracy * 100).toFixed(2)}%`);
        }
      } catch (e) {
        console.warn('Could not load MLP model.json:', e);
      }

      // 2. Load labels if not already populated from model
      if (!this.labels || this.labels.length === 0) {
        try {
          const labelsRes = await fetch(`${modelDir}/labels.json${cacheBust}`);
          if (labelsRes.ok) {
            this.labels = await labelsRes.json();
            console.log(`✓ Loaded ${this.labels.length} labels from labels.json`);
          }
        } catch (e) {
          console.warn('Could not load labels.json:', e);
        }
      }

      // 3. Load KNN Centroids as fallback
      try {
        const knnRes = await fetch(`${modelDir}/landmarks.json${cacheBust}`);
        if (knnRes.ok) {
          const knnData = await knnRes.json();
          this.knnCentroids = knnData.centroids;
          console.log('✓ Loaded KNN reference centroids.');
        }
      } catch (e) {
        console.warn('Could not load landmarks.json:', e);
      }

      if (!this.model && !this.knnCentroids) {
        throw new Error('Neither MLP model nor KNN centroids could be loaded.');
      }

      this.initBuffers();
      this.isLoaded = true;
      return true;
    } catch (err) {
      console.error('Error loading classifier artifacts:', err);
      return false;
    }
  }

  /**
   * Rotates 21 landmark points {x, y, z} in the XY plane so that the vector from
   * Landmark 0 (wrist) to Landmark 9 (middle MCP) points vertically UP (angle = -90°).
   */
  normalizeLandmarks(landmarks) {
    if (!landmarks || landmarks.length !== 21) {
      throw new Error(`Expected 21 landmarks, received: ${landmarks ? landmarks.length : 0}`);
    }

    const wrist = landmarks[0];
    const midMcp = landmarks[9];

    // Vector from wrist to middle MCP
    const dx = midMcp.x - wrist.x;
    const dy = midMcp.y - wrist.y;
    const angle = Math.atan2(dy, dx);

    // Target is -PI/2 (-90 deg: straight up in image coordinates where y=0 is top)
    const rotAngle = -Math.PI / 2.0 - angle;
    const cosT = Math.cos(rotAngle);
    const sinT = Math.sin(rotAngle);
    const wristZ = wrist.z || 0;

    let maxDist = 0;

    for (let i = 0; i < 21; i++) {
      const rx = landmarks[i].x - wrist.x;
      const ry = landmarks[i].y - wrist.y;
      const rz = (landmarks[i].z !== undefined && !isNaN(landmarks[i].z))
        ? (landmarks[i].z - wristZ)
        : 0;

      // 2D Rotation around wrist
      const rotX = rx * cosT - ry * sinT;
      const rotY = rx * sinT + ry * cosT;

      const rotLm = this.rotLandmarks[i];
      rotLm.x = rotX;
      rotLm.y = rotY;
      rotLm.z = rz;

      const d = Math.sqrt(rotX * rotX + rotY * rotY + rz * rz);
      if (d > maxDist) {
        maxDist = d;
      }
    }

    if (maxDist < 1e-6) {
      maxDist = 1.0;
    }

    const vec = this.normVector;
    for (let i = 0; i < 21; i++) {
      const rotLm = this.rotLandmarks[i];
      vec[i * 3 + 0] = rotLm.x / maxDist;
      vec[i * 3 + 1] = rotLm.y / maxDist;
      vec[i * 3 + 2] = rotLm.z / maxDist;
    }

    return vec;
  }

  /**
   * Generic forward pass for MLP Neural Network with zero heap allocation.
   */
  predictMLP(vector) {
    const weights = this.model.weights;
    const biases = this.model.biases;
    const numLayers = weights.length;

    let currentActivation = vector;

    for (let l = 0; l < numLayers; l++) {
      const W = weights[l];
      const b = biases[l];
      const inDim = currentActivation.length;
      const outDim = b.length;
      const nextActivation = this.layerBuffers[l];
      const isOutputLayer = (l === numLayers - 1);

      for (let j = 0; j < outDim; j++) {
        let sum = b[j];
        for (let i = 0; i < inDim; i++) {
          sum += currentActivation[i] * W[i][j];
        }
        nextActivation[j] = isOutputLayer ? sum : (sum > 0 ? sum : 0);
      }
      currentActivation = nextActivation;
    }

    // Softmax directly into this.rawProbs
    const logits = currentActivation;
    let maxLogit = -Infinity;
    for (let j = 0; j < logits.length; j++) {
      if (logits[j] > maxLogit) maxLogit = logits[j];
    }

    let sumExp = 0;
    const raw = this.rawProbs;
    for (let j = 0; j < logits.length; j++) {
      const e = Math.exp(logits[j] - maxLogit);
      raw[j] = e;
      sumExp += e;
    }
    for (let j = 0; j < logits.length; j++) {
      raw[j] /= sumExp;
    }

    return raw;
  }

  /**
   * Deterministic Geometric ASL Disambiguation:
   * Provides high-precision physical disambiguation for closely related sign clusters:
   * 1. C vs O (Open cup vs tightly pinched circle)
   * 2. K vs V vs P vs U vs H vs R (Index & Middle 2-finger variations)
   * 3. Q vs G vs P vs L (Index & Thumb variations / Downward pointing)
   * 4. Fist Cluster: A, E, S, T, M, N (Thumb tucked / drape positions)
   * 5. Hooked Index X (Only when index PIP is high above fist and tip bent down)
   */
  evaluateGeometry(landmarks, rawProbs) {
    const wrist = landmarks[0];
    const midMcp = landmarks[9];

    const dx = midMcp.x - wrist.x;
    const dy = midMcp.y - wrist.y;
    const palmLen = Math.hypot(dx, dy) || 0.1;
    const rawAngleDeg = (Math.atan2(dy, dx) * 180) / Math.PI;

    // Use pre-rotated landmarks from normalizeLandmarks (zero allocations)
    const rot = this.rotLandmarks;

    const d = (i, j) => {
      const dX = rot[i].x - rot[j].x;
      const dY = rot[i].y - rot[j].y;
      const dZ = rot[i].z - rot[j].z;
      return Math.sqrt(dX * dX + dY * dY + dZ * dZ);
    };

    const d2 = (i, j) => Math.hypot(rot[i].x - rot[j].x, rot[i].y - rot[j].y);

    // Finger extensions in rotated coordinates (Y is negative UPWARDS)
    const isExt = (tip, pip) => (rot[tip].y < rot[pip].y - 0.05 * palmLen) && (d(0, tip) > d(0, pip) * 1.02);

    const idxExt = isExt(8, 6);
    const midExt = isExt(12, 10);
    const ringExt = isExt(16, 14);
    const pinkyExt = isExt(20, 18);

    const idxCurl = !idxExt;
    const midCurl = !midExt;
    const ringCurl = !ringExt;
    const pinkyCurl = !pinkyExt;

    // Relative distance of fingertips from palm base
    const dBase = (tip) => d(0, tip) / palmLen;
    const dThumbIdx = d(4, 8) / palmLen;
    const dThumbMid = d(4, 12) / palmLen;

    // Hand orientation in camera frame: dy > 0 in pixel space means pointing DOWNWARDS
    const isPointingDown = (rawAngleDeg > 35) && (rawAngleDeg < 145);

    // Inter-finger distances & crossings
    const dIdxMid = d2(8, 12) / palmLen;
    const crossedIdxMid = ((rot[8].x - rot[12].x) * (rot[5].x - rot[9].x)) < 0;

    // Find top-1 from raw MLP probabilities if available
    let top1Label = '';
    if (rawProbs && this.labels.length > 0) {
      let maxP = -1, maxI = 0;
      for (let i = 0; i < rawProbs.length; i++) {
        if (rawProbs[i] > maxP) { maxP = rawProbs[i]; maxI = i; }
      }
      top1Label = this.labels[maxI] || '';
    }

    // 1. Hooked Index -> 'X' ONLY if index PIP is genuinely raised high above palm,
    // only the tip is curled down, and other 3 fingers are folded deep in fist
    const idxPipUp = (rot[6].y < rot[5].y - 0.15 * palmLen) && (dBase(6) > 0.65);
    const idxTipDown = rot[8].y > rot[6].y + 0.04 * palmLen;
    const otherFingersFolded = (dBase(10) < 0.65) && (dBase(14) < 0.60) && (dBase(18) < 0.55);
    if (idxPipUp && idxTipDown && otherFingersFolded) {
      return { letter: 'X', boost: 0.45 };
    }

    // 2. 4 Fingers Extended -> 'B' (straight flat palm)
    if (idxExt && midExt && ringExt && pinkyExt && dBase(8) > 0.85 && dBase(12) > 0.85) {
      return { letter: 'B', boost: 0.40 };
    }

    // 3. 3 Fingers Extended (Middle, Ring, Pinky) + Thumb touching Index -> 'F'
    if (midExt && ringExt && pinkyExt && dThumbIdx < 0.28 && idxCurl) {
      return { letter: 'F', boost: 0.45 };
    }

    // 4. 3 Fingers Extended (Index, Middle, Ring) -> 'W'
    if (idxExt && midExt && ringExt && pinkyCurl) {
      return { letter: 'W', boost: 0.40 };
    }

    // 5. CLUSTER C vs O
    if (top1Label === 'C' || top1Label === 'O' || (idxCurl && midCurl && ringCurl && pinkyCurl && dBase(8) > 0.40)) {
      // O: tight closed circle where thumb tip pinches index and middle fingertips
      const isO = (dThumbIdx < 0.22) && (dThumbMid < 0.25);
      // C: open rounded cup with distinct visible gap between thumb and fingertips
      const isC = (dThumbIdx > 0.28) && (dBase(8) > 0.50);
      if (isO) return { letter: 'O', boost: 0.45 };
      if (isC) return { letter: 'C', boost: 0.45 };
    }

    // 6. CLUSTER K vs V vs P vs U vs H vs R (2 fingers extended: Index & Middle)
    if (top1Label === 'K' || top1Label === 'V' || top1Label === 'P' || top1Label === 'U' || top1Label === 'H' || top1Label === 'R' || (idxExt && midExt && ringCurl && pinkyCurl)) {
      if (isPointingDown) {
        return { letter: 'P', boost: 0.45 };
      }
      if (crossedIdxMid) {
        return { letter: 'R', boost: 0.45 };
      }
      // K: thumb tip placed upwards between index and middle
      const thumbUpBetween = (rot[4].y < rot[2].y) &&
                             (rot[4].y < rot[6].y + 0.05 * palmLen) &&
                             (d(4, 9) < 0.40 * palmLen || d(4, 5) < 0.40 * palmLen);
      if (thumbUpBetween) {
        return { letter: 'K', boost: 0.40 };
      }
      if (dIdxMid > 0.28) {
        return { letter: 'V', boost: 0.35 };
      }
      return { letter: 'U', boost: 0.35 };
    }

    // 7. CLUSTER Q vs G vs P vs L (Index extended or Index + Thumb)
    if (top1Label === 'Q' || top1Label === 'G' || top1Label === 'L' || (idxExt && midCurl && ringCurl && pinkyCurl)) {
      if (isPointingDown) {
        return { letter: 'Q', boost: 0.45 };
      }
      const thumbOut = Math.abs(rot[4].x - rot[9].x) > 0.40 * palmLen;
      if (thumbOut && rot[4].y < rot[2].y - 0.05 * palmLen && rot[8].y < rot[6].y - 0.15 * palmLen) {
        return { letter: 'L', boost: 0.40 };
      }
      if (thumbOut) {
        return { letter: 'G', boost: 0.35 };
      }
    }

    // 8. 2 Fingers Extended (Thumb & Pinky) -> 'Y'
    const thumbOut = Math.abs(rot[4].x - rot[9].x) > 0.40 * palmLen;
    if (thumbOut && pinkyExt && idxCurl && midCurl && ringCurl) {
      return { letter: 'Y', boost: 0.45 };
    }

    // 9. 1 Finger Extended (Pinky only) -> 'I'
    if (pinkyExt && idxCurl && midCurl && ringCurl && !thumbOut) {
      return { letter: 'I', boost: 0.45 };
    }

    // 10. CLUSTER FIST FAMILY: A, E, S, T, M, N
    if (idxCurl && midCurl && ringCurl && pinkyCurl && (dBase(8) < 0.65)) {
      // A: thumb pointing upright along the side of index MCP (pointing up)
      const thumbUpright = (rot[4].y < rot[2].y - 0.05 * palmLen) &&
                           (d(4, 5) < 0.42 * palmLen) &&
                           (rot[4].y < rot[8].y - 0.03 * palmLen);
      if (thumbUpright) return { letter: 'A', boost: 0.40 };

      // E: thumb tucked under curled fingertips forming a horizontal shelf
      const thumbUnderFingertips = (rot[4].y >= rot[8].y - 0.04 * palmLen) &&
                                  (d(4, 8) < 0.35 * palmLen || d(4, 12) < 0.35 * palmLen);
      if (thumbUnderFingertips) return { letter: 'E', boost: 0.40 };

      // S: thumb folded horizontally ACROSS the front of index/middle fingers
      const thumbFoldedAcross = (rot[4].y < rot[8].y) &&
                                (d(4, 9) < 0.38 * palmLen) &&
                                (rot[4].y > rot[5].y - 0.06 * palmLen);
      if (thumbFoldedAcross) return { letter: 'S', boost: 0.40 };

      // T / N / M: sub-classify based on thumb tip position relative to knuckles
      const thumbTipX = rot[4].x;
      const idxKnuckleX = rot[5].x;
      const midKnuckleX = rot[9].x;
      const ringKnuckleX = rot[13].x;
      const pinkyKnuckleX = rot[17].x;

      if (d(4, 9) < 0.55 * palmLen || d(4, 5) < 0.55 * palmLen) {
        if (Math.abs(thumbTipX - idxKnuckleX) < 0.25 * palmLen && d(4, 5) < 0.40 * palmLen) {
          return { letter: 'T', boost: 0.35 };
        }
        if (d(4, 9) < 0.40 * palmLen && Math.abs(thumbTipX - midKnuckleX) < 0.25 * palmLen) {
          return { letter: 'N', boost: 0.35 };
        }
        if (d(4, 13) < 0.45 * palmLen || d(4, 17) < 0.45 * palmLen) {
          return { letter: 'M', boost: 0.35 };
        }
      }
    }

    return null;
  }

  /**
   * Main prediction entry point given raw MediaPipe landmarks.
   */
  predict(landmarks) {
    if (!this.isLoaded) {
      return { label: '...', confidence: 0, top3: [], isLowConfidence: true };
    }

    // 1. Normalization
    const normVector = this.normalizeLandmarks(landmarks);

    // 2. MLP Evaluation
    let mlpProbs = null;
    if (this.model) {
      mlpProbs = this.predictMLP(normVector);
    }

    const numClasses = this.labels.length;
    const finalProbs = this.finalProbs;

    if (mlpProbs) {
      for (let j = 0; j < numClasses; j++) {
        finalProbs[j] = mlpProbs[j];
      }
    } else if (this.knnCentroids) {
      for (let j = 0; j < numClasses; j++) {
        const c = this.knnCentroids[this.labels[j]];
        if (c) {
          let sumSq = 0;
          for (let k = 0; k < 63; k++) {
            const diff = c[k] - normVector[k];
            sumSq += diff * diff;
          }
          finalProbs[j] = Math.exp(-Math.sqrt(sumSq) * 3.0);
        }
      }
    }

    // 3. Deterministic Geometric ASL Disambiguation (Soft Additive Boost)
    const geomMatch = this.evaluateGeometry(landmarks, finalProbs);
    if (geomMatch) {
      const matchIdx = this.labels.indexOf(geomMatch.letter);
      if (matchIdx !== -1) {
        finalProbs[matchIdx] += (geomMatch.boost || 0.40);
      }
    }

    // Normalize probabilities to sum to 1
    let sumP = 0;
    for (let j = 0; j < numClasses; j++) sumP += finalProbs[j];
    if (sumP > 1e-6) {
      for (let j = 0; j < numClasses; j++) finalProbs[j] /= sumP;
    }

    // Find top-1 candidate
    let topIdx = 0;
    let topProb = -1;
    for (let j = 0; j < numClasses; j++) {
      if (finalProbs[j] > topProb) {
        topProb = finalProbs[j];
        topIdx = j;
      }
    }

    const currentCandidate = this.labels[topIdx];

    // Reset EMA smoothing when candidate changes or buffer empty
    if (this.lastPredictedLabel !== currentCandidate) {
      for (let j = 0; j < numClasses; j++) {
        this.emaProbabilities[j] = finalProbs[j];
      }
      this.lastPredictedLabel = currentCandidate;
    } else {
      const alpha = this.emaAlpha;
      const invAlpha = 1 - alpha;
      for (let j = 0; j < numClasses; j++) {
        this.emaProbabilities[j] = alpha * finalProbs[j] + invAlpha * this.emaProbabilities[j];
      }
    }

    // Update indexedProbs in-place and sort
    const indexed = this.indexedProbs;
    for (let j = 0; j < numClasses; j++) {
      indexed[j].prob = this.emaProbabilities[j];
      indexed[j].label = this.labels[j];
      indexed[j].index = j;
    }
    indexed.sort((a, b) => b.prob - a.prob);

    const top1 = indexed[0];
    const top2 = indexed[1];
    const top3 = indexed[2];

    // Require high confidence (0.48+) to accept prediction
    const isLowConfidence = top1.prob < 0.48;

    return {
      label: isLowConfidence ? '—' : top1.label,
      confidence: top1.prob,
      index: top1.index,
      top3: [
        { label: top1.label, confidence: top1.prob, index: top1.index },
        { label: top2.label, confidence: top2.prob, index: top2.index },
        { label: top3.label, confidence: top3.prob, index: top3.index }
      ],
      isLowConfidence: isLowConfidence
    };
  }

  /**
   * Resets smoothing buffer (call when hand is lost or during cooldown).
   */
  resetSmoothing() {
    this.lastPredictedLabel = null;
    if (this.emaProbabilities) {
      this.emaProbabilities.fill(0);
    }
  }
}

window.SignClassifier = SignClassifier;
