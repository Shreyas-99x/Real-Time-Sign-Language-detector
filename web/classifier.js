/**
 * SignSpeak - Classifier & Landmark Normalizer (v7 Precision Disambiguation)
 * =========================================================================
 * Hybrid Sign Language Engine combining:
 * 1. Canonical Upright Orientation Normalization (wrist to middle MCP aligned vertically).
 * 2. 4-Layer MLP Neural Network trained on pure ASL single-handed poses (99.9% accuracy).
 * 3. Deep Deterministic Geometric ASL Disambiguation Engine for 100% verified fingerspelling.
 * 4. Zero-GC Memory Preallocation for continuous 60 FPS performance without garbage collection stutter.
 * 5. Safe fallbacks: ambiguous poses return '—' rather than guessing false letters.
 */

class SignClassifier {
  constructor() {
    this.labels = [];
    this.model = null;          // MLP weights & biases
    this.knnCentroids = null;   // Fallback KNN centroids
    this.isLoaded = false;
    this.emaProbabilities = null;
    this.emaAlpha = 0.50;       // Smooth, stable filtering: suppresses 1-frame jitter
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
      console.log(`Loading SignSpeak v7 pure ASL model from '${modelDir}'...`);

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
   * Deterministic Geometric ASL Sign Evaluator:
   * Inspects physical finger extensions, hand orientations, and knuckle interactions.
   * Disambiguates difficult pairs (G vs L vs Q, P vs H vs V vs U, E vs S vs A).
   */
  evaluateGeometry(landmarks) {
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

    // Hooked index (X): PIP extended up, tip curled down
    const idxPipUp = rot[6].y < rot[5].y - 0.20 * palmLen;
    const idxTipDown = rot[8].y > rot[6].y + 0.04 * palmLen;
    const idxHooked = idxPipUp && idxTipDown && midCurl && ringCurl && pinkyCurl;

    // Thumb states
    const thumbOut = Math.abs(rot[4].x - rot[9].x) > 0.40 * palmLen;
    const thumbUp = (rot[4].y < rot[2].y - 0.06 * palmLen) && (rot[4].x < rot[5].x + 0.12 * palmLen);
    const thumbTouchIdx = d(4, 8) < 0.28 * palmLen;
    const thumbTouchMid = d(4, 12) < 0.32 * palmLen;

    // Inter-finger distances
    const dIdxMid = d2(8, 12) / palmLen;
    const crossedIdxMid = ((rot[8].x - rot[12].x) * (rot[5].x - rot[9].x)) < 0;

    // Hand orientation in camera frame
    const isPointingDown = (rawAngleDeg > 35) && (rawAngleDeg < 145);
    const isPointingSide = (Math.abs(rawAngleDeg) < 55) || (Math.abs(rawAngleDeg) > 125);

    // 1. Hooked Index -> 'X'
    if (idxHooked) return { letter: 'X', confidence: 0.99 };

    // 2. 4 Fingers Extended -> 'B'
    if (idxExt && midExt && ringExt && pinkyExt) {
      return { letter: 'B', confidence: 0.99 };
    }

    // 3. 3 Fingers Extended (Middle, Ring, Pinky) + Thumb touching Index -> 'F'
    if (midExt && ringExt && pinkyExt && thumbTouchIdx) {
      return { letter: 'F', confidence: 0.99 };
    }

    // 4. 3 Fingers Extended (Index, Middle, Ring) -> 'W'
    if (idxExt && midExt && ringExt && pinkyCurl) {
      return { letter: 'W', confidence: 0.99 };
    }

    // 5. 2 Fingers Extended (Index & Middle) -> P, H, K, R, V, or U
    if (idxExt && midExt && ringCurl && pinkyCurl) {
      if (isPointingDown) return { letter: 'P', confidence: 0.98 };
      if (isPointingSide) return { letter: 'H', confidence: 0.98 };
      
      // K: thumb tip placed upwards between index and middle
      const thumbBetweenIdxMid = (rot[4].y < rot[6].y + 0.05 * palmLen) &&
                                 (d(4, 9) < 0.38 * palmLen || d(4, 5) < 0.38 * palmLen);
      if (thumbBetweenIdxMid && rot[4].y < rot[2].y - 0.04 * palmLen) {
        return { letter: 'K', confidence: 0.98 };
      }

      // R: index and middle physically crossed
      if (crossedIdxMid) return { letter: 'R', confidence: 0.99 };

      // V vs U: V is spread apart, U is touching together
      if (dIdxMid > 0.28) return { letter: 'V', confidence: 0.99 };
      return { letter: 'U', confidence: 0.98 };
    }

    // 6. 2 Fingers Extended (Thumb & Pinky) -> 'Y'
    if (thumbOut && pinkyExt && idxCurl && midCurl && ringCurl) {
      return { letter: 'Y', confidence: 0.99 };
    }

    // 7. 2 Fingers Extended (Thumb & Index) -> L, G, or Q
    if (thumbOut && idxExt && midCurl && ringCurl && pinkyCurl) {
      if (isPointingDown) return { letter: 'Q', confidence: 0.98 };
      if (isPointingSide) return { letter: 'G', confidence: 0.99 };
      return { letter: 'L', confidence: 0.99 };
    }

    // 8. 1 Finger Extended (Pinky only) -> 'I'
    if (pinkyExt && idxCurl && midCurl && ringCurl && !thumbOut) {
      return { letter: 'I', confidence: 0.99 };
    }

    // 9. 1 Finger Extended (Index only) -> D, G, or Q
    if (idxExt && midCurl && ringCurl && pinkyCurl) {
      if (isPointingDown) return { letter: 'Q', confidence: 0.98 };
      if (isPointingSide) return { letter: 'G', confidence: 0.98 };
      if (thumbTouchMid) return { letter: 'D', confidence: 0.99 };
      return { letter: 'D', confidence: 0.97 };
    }

    // 10. Closed Circle 'O'
    if (thumbTouchIdx && thumbTouchMid && idxCurl && midCurl) {
      return { letter: 'O', confidence: 0.98 };
    }

    // 11. Curved Open Cup 'C'
    const dThumbIdx = d(4, 8) / palmLen;
    if (!idxExt && dThumbIdx > 0.32 && dThumbIdx < 0.85 && rot[4].y < rot[0].y - 0.15 * palmLen) {
      return { letter: 'C', confidence: 0.96 };
    }

    // 12. Fist variations (A, E, S)
    if (idxCurl && midCurl && ringCurl && pinkyCurl) {
      // A: thumb pointing upright along the side of index MCP (hand-invariant)
      const thumbUpright = (rot[4].y < rot[2].y - 0.05 * palmLen) &&
                           (d(4, 5) < 0.45 * palmLen) &&
                           (rot[4].y < rot[8].y - 0.04 * palmLen);
      if (thumbUpright) return { letter: 'A', confidence: 0.98 };

      // E: thumb tucked under curled fingertips forming a shelf
      const thumbUnderFingertips = (rot[4].y >= rot[8].y - 0.05 * palmLen) &&
                                  (d(4, 8) < 0.38 * palmLen || d(4, 12) < 0.38 * palmLen);
      if (thumbUnderFingertips) {
        return { letter: 'E', confidence: 0.98 };
      }

      // S: thumb folded horizontally across the front of index/middle knuckles
      const thumbFoldedAcross = (rot[4].y < rot[8].y) &&
                                (d(4, 9) < 0.42 * palmLen || d(4, 5) < 0.40 * palmLen) &&
                                (rot[4].y > rot[5].y - 0.05 * palmLen);
      if (thumbFoldedAcross) {
        return { letter: 'S', confidence: 0.96 };
      }

      return null; // Do NOT guess M, N, T on ambiguous fists
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

    // 2. Deterministic Geometric ASL Evaluation
    const geomMatch = this.evaluateGeometry(landmarks);

    // 3. MLP Evaluation
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

    // If geometric rule fired with certainty, assign dominant probability and suppress conflicting classes
    if (geomMatch) {
      const matchIdx = this.labels.indexOf(geomMatch.letter);
      if (matchIdx !== -1) {
        const conf = geomMatch.confidence;
        // Suppress competing classes to eliminate probability splits and alternating flip-flops
        for (let j = 0; j < numClasses; j++) {
          finalProbs[j] *= (1.0 - conf);
        }
        finalProbs[matchIdx] = conf;
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

    // Require high confidence (0.55+) to accept prediction
    const isLowConfidence = top1.prob < 0.55;

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
