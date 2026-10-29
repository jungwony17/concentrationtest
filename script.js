'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const playfield = $('playfield');
  const target = $('target');
  const shadow = $('targetShadow');
  const reticle = $('gazeReticle');
  const centerInstruction = $('centerInstruction');
  const startOverlay = $('startOverlay');
  const calibrationOverlay = $('calibrationOverlay');
  const resultOverlay = $('resultOverlay');
  const webcamStartButton = $('webcamStartButton');
  const touchStartButton = $('touchStartButton');
  const hybridStartButton = $('hybridStartButton');
  const cancelCalibrationButton = $('cancelCalibrationButton');
  const restartButton = $('restartButton');
  const libraryStatus = $('libraryStatus');
  const statusText = $('statusText');
  const modePill = $('modePill');
  const accuracyText = $('accuracyText');
  const accuracyBar = $('accuracyBar');
  const accuracyTrack = $('accuracyTrack');
  const timerText = $('timerText');
  const calibrationPoints = $('calibrationPoints');
  const calibrationCount = $('calibrationCount');
  const calibrationProgress = $('calibrationProgress');
  const saveStatus = $('saveStatus');
  const participantCodeInput = $('participantCode');

  const TEST_SECONDS = 60;
  const TARGET_MOVE_SECONDS = 14; // 목표 공은 모든 테스트에서 14초 동안 이동합니다.
  const SAMPLE_INTERVAL_MS = 150;
  const PHASE_SECONDS = 20; // 초반/중반/후반은 각 20초 구간입니다.
  const CALIBRATION_POSITIONS = [[8, 10], [50, 10], [92, 10], [8, 50], [50, 50], [92, 50], [8, 90], [50, 90], [92, 90]];
  const FACE_MESH_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/';
  // Apps Script 웹 앱 배포 후 URL을 여기에 붙여넣으세요. 비워두면 관리자 시트 저장은 꺼져 있습니다.
  const ADMIN_SHEETS_WEB_APP_URL = 'https://script.google.com/macros/s/AKfycbz4Y7rEVyGOkcA8O2L6doM8RFAyWqsZO_9jy-EdYMakmwmo64gFNEYojI-pC1Ooy7qXSA/exec';
  const GAZE_FILTER_VERSION = 'median5-ema-adaptive-v3';
  const CALIBRATION_CLICKS_PER_POINT = 5;

  let mode = 'webcam';
  let running = false;
  let gazeReady = false;
  let sampleTimer = null;
  let gameClock = null;
  let animationFrame = null;
  let remainingSeconds = TEST_SECONDS;
  let totalSamples = 0;
  let hitSamples = 0;
  let latestPoint = null;
  let latestPointTime = 0;
  let latestMousePoint = null;
  let latestMousePointTime = 0;
  let mouseInside = false;
  let mouseTotalSamples = 0;
  let mouseHitSamples = 0;
  let phaseStats = [newPhaseStat(), newPhaseStat(), newPhaseStat()];
  let mousePhaseStats = [newPhaseStat(), newPhaseStat(), newPhaseStat()];
  let webcamGameMode = 'webcam';
  let calibrationIndex = 0;
  let calibrationClicks = 0;
  let calibrationEls = [];
  let currentPoint = { x: 50, y: 50 };
  let nextPoint = { x: 50, y: 50 };
  let segmentStart = 0;
  let touchActive = false;
  let pointerInside = false;
  let lastPointerType = 'mouse';
  let gazeBuffer = [];
  let filteredGaze = null;
  let sessionStartedMs = 0;
  let currentSessionId = '';

  function show(element) { element.classList.remove('hidden'); }
  function hide(element) { element.classList.add('hidden'); }
  function setStatus(text, active = false) {
    statusText.textContent = text;
    playfield.classList.toggle('playing', active);
    modePill.textContent = active
      ? (mode === 'webcam' ? '시선 추적 중' : mode === 'hybrid' ? '시선 + 마우스 추적 중' : '터치 체험 중')
      : text;
  }
  function setLibraryMessage(text, error = false) {
    libraryStatus.textContent = text;
    libraryStatus.classList.toggle('error', error);
  }
  function formatTime(seconds) {
    const s = Math.max(0, seconds);
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }
  function updateAccuracy() {
    const percent = totalSamples ? Math.round((hitSamples / totalSamples) * 100) : 0;
    accuracyText.innerHTML = totalSamples ? `${percent}<small>%</small>` : `--<small>%</small>`;
    accuracyBar.style.width = `${percent}%`;
    accuracyTrack.setAttribute('aria-valuenow', String(percent));
  }
  function viewportPointToField(x, y) {
    const rect = playfield.getBoundingClientRect();
    reticle.style.left = `${x - rect.left}px`;
    reticle.style.top = `${y - rect.top}px`;
    reticle.classList.add('visible');
  }
  function updateLatestPoint(x, y, source, pointerType = 'mouse') {
    latestPoint = { x, y, source, pointerType };
    latestPointTime = Date.now();
    viewportPointToField(x, y);
  }
  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  // 웹캠 시선 좌표의 순간 튐은 중앙값으로 줄이되, 이전 버전의 지나친 지연을 완화합니다.
  // 최근 5개 좌표의 중앙값 + 이동량에 따라 달라지는 EMA + 작은 데드밴드를 사용합니다.
  function onGazePrediction(data) {
    if (!data || !Number.isFinite(data.x) || !Number.isFinite(data.y)) return;
    const now = Date.now();
    const x = data.x;
    const y = data.y;
    if (x < -80 || y < -80 || x > window.innerWidth + 80 || y > window.innerHeight + 80) return;

    // 예측이 1.2초 넘게 끊긴 경우엔 오래된 필터 상태를 버리고 다시 시작합니다.
    if (!filteredGaze || now - latestPointTime > 1200) {
      gazeBuffer = [{ x, y }];
      filteredGaze = { x, y };
      updateLatestPoint(filteredGaze.x, filteredGaze.y, 'webcam');
      return;
    }

    gazeBuffer.push({ x, y });
    if (gazeBuffer.length > 5) gazeBuffer.shift();
    const mx = median(gazeBuffer.map((point) => point.x));
    const my = median(gazeBuffer.map((point) => point.y));
    const dx = mx - filteredGaze.x;
    const dy = my - filteredGaze.y;
    const distance = Math.hypot(dx, dy);

    // 포인터가 가만히 있어야 하는 상황의 미세한 떨림을 줄입니다.
    if (distance >= 5) {
      let alpha;
      // 작은 떨림은 완화하고, 실제 시선 이동은 이전 버전보다 빠르게 반영합니다.
      if (distance < 30) alpha = 0.20;
      else if (distance < 70) alpha = 0.30;
      else if (distance < 140) alpha = 0.42;
      else alpha = 0.52;

      let stepX = dx * alpha;
      let stepY = dy * alpha;
      const step = Math.hypot(stepX, stepY);
      const maxStep = 64;
      if (step > maxStep) {
        const scale = maxStep / step;
        stepX *= scale;
        stepY *= scale;
      }
      filteredGaze.x += stepX;
      filteredGaze.y += stepY;
    }
    // 포인터 좌표가 그대로여도 최신 예측 시각은 갱신합니다.
    updateLatestPoint(filteredGaze.x, filteredGaze.y, 'webcam');
  }

  function randomWaypoint(from) {
    // 화면 가장자리의 지나치게 좁은 구역을 피해서 부드럽게 이동합니다.
    let point;
    let tries = 0;
    do {
      point = { x: 10 + Math.random() * 80, y: 13 + Math.random() * 74 };
      tries += 1;
    } while (tries < 8 && Math.hypot(point.x - from.x, point.y - from.y) < 24);
    return point;
  }
  function renderTarget(x, y) {
    target.style.left = `${x}%`;
    target.style.top = `${y}%`;
    shadow.style.left = `${x}%`;
    shadow.style.top = `${y}%`;
  }
  function tickTarget(now) {
    if (!running) return;
    if (!segmentStart) {
      segmentStart = now;
      currentPoint = { ...nextPoint };
      nextPoint = randomWaypoint(currentPoint);
    }
    const moveSeconds = TARGET_MOVE_SECONDS;
    const rawT = Math.min(1, (now - segmentStart) / (moveSeconds * 1000));
    // smoothstep keeps both the start and end of each journey gentle rather than teleporting.
    const t = rawT * rawT * (3 - 2 * rawT);
    const x = currentPoint.x + (nextPoint.x - currentPoint.x) * t;
    const y = currentPoint.y + (nextPoint.y - currentPoint.y) * t;
    renderTarget(x, y);
    if (rawT >= 1) {
      currentPoint = { ...nextPoint };
      nextPoint = randomWaypoint(currentPoint);
      segmentStart = now;
    }
    animationFrame = requestAnimationFrame(tickTarget);
  }
  function startTargetAnimation() {
    cancelAnimationFrame(animationFrame);
    segmentStart = 0;
    currentPoint = { x: 50, y: 50 };
    nextPoint = { x: 50, y: 50 };
    renderTarget(50, 50);
    animationFrame = requestAnimationFrame(tickTarget);
  }

  async function startWebcamFlow(nextMode = 'webcam') {
    if (!window.isSecureContext) {
      setLibraryMessage('휴대폰에서 웹캠을 쓰려면 HTTPS 주소로 접속해야 해요. PC에서는 localhost로 테스트할 수 있습니다.', true);
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setLibraryMessage('이 브라우저에서 웹캠 접근을 지원하지 않아요. 최신 Chrome 또는 Safari에서 다시 시도하거나 터치 모드를 이용해 주세요.', true);
      return;
    }
    const wg = window.webgazer;
    if (!wg) {
      setLibraryMessage('시선 추적 라이브러리를 불러오지 못했어요. 인터넷 연결을 확인한 뒤 새로고침해 주세요.', true);
      return;
    }
    webcamStartButton.disabled = true;
    hybridStartButton.disabled = true;
    webcamStartButton.textContent = '웹캠 연결 중…';
    setLibraryMessage('카메라와 시선 추적 모델을 연결하고 있어요…');
    try {
      // Important: use the CDN-hosted library build plus absolute MediaPipe model path.
      // Do this before begin(), when the face-mesh tracker is created.
      gazeBuffer = [];
      filteredGaze = null;
      if (wg.params) {
        wg.params.faceMeshSolutionPath = FACE_MESH_CDN;
        wg.params.saveDataAcrossSessions = false;
        wg.params.camConstraints = {
          video: { facingMode: { ideal: 'user' }, width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false
        };
      }
      if (typeof wg.showVideoPreview === 'function') wg.showVideoPreview(false);
      if (typeof wg.showPredictionPoints === 'function') wg.showPredictionPoints(false);
      if (typeof wg.showFaceOverlay === 'function') wg.showFaceOverlay(false);
      if (typeof wg.showFaceFeedbackBox === 'function') wg.showFaceFeedbackBox(false);
      if (typeof wg.setGazeListener === 'function') wg.setGazeListener(onGazePrediction);
      await wg.begin();
      gazeReady = true;
      mode = nextMode;
      webcamGameMode = nextMode;
      hide(startOverlay);
      beginCalibration();
      setStatus('웹캠 연결됨 · 보정 중', true);
      setLibraryMessage(nextMode === 'hybrid'
        ? '연결되었어요. 보정 후 공을 눈으로 따라가고, 마우스도 움직이면 두 기록을 비교해요.'
        : '연결되었어요. 보정 점 9개를 차례로 눌러 주세요.');
    } catch (error) {
      console.error('시선 추적 시작 오류:', error);
      const name = error && error.name ? `${error.name}: ` : '';
      const message = error && error.message ? error.message : '알 수 없는 오류';
      setLibraryMessage(`${name}${message} — 카메라 권한, 인터넷 연결, 브라우저 호환성을 확인해 주세요. 모바일에서 계속 실패하면 터치 모드로 테스트할 수 있어요.`, true);
      webcamStartButton.disabled = false;
      webcamStartButton.innerHTML = '<span class="button-icon">◎</span> 다시 웹캠 연결하기 <span class="button-arrow">↻</span>';
      hybridStartButton.disabled = false;
    }
  }

  function beginCalibration() {
    calibrationIndex = 0;
    calibrationClicks = 0;
    calibrationEls = [];
    calibrationPoints.innerHTML = '';
    CALIBRATION_POSITIONS.forEach(([x, y], index) => {
      const point = document.createElement('button');
      point.type = 'button';
      point.className = `calibration-point${index === 0 ? ' current' : ''}`;
      point.style.left = `${x}%`;
      point.style.top = `${y}%`;
      point.setAttribute('aria-label', `시선 보정 지점 ${index + 1}`);
      point.addEventListener('click', () => onCalibrationPointClick(index, point));
      calibrationPoints.appendChild(point);
      calibrationEls.push(point);
    });
    updateCalibrationProgress();
    show(calibrationOverlay);
  }
  function updateCalibrationProgress() {
    const shownPoint = Math.min(calibrationIndex + 1, CALIBRATION_POSITIONS.length);
    calibrationCount.textContent = calibrationIndex >= CALIBRATION_POSITIONS.length
      ? `${CALIBRATION_POSITIONS.length} / ${CALIBRATION_POSITIONS.length}`
      : `${shownPoint} / ${CALIBRATION_POSITIONS.length} · ${calibrationClicks}/${CALIBRATION_CLICKS_PER_POINT} 클릭`;
    calibrationProgress.style.width = `${((calibrationIndex + calibrationClicks / CALIBRATION_CLICKS_PER_POINT) / CALIBRATION_POSITIONS.length) * 100}%`;
    calibrationEls.forEach((point, index) => {
      point.classList.toggle('done', index < calibrationIndex);
      point.classList.toggle('current', index === calibrationIndex);
      point.disabled = index !== calibrationIndex;
    });
  }
  function onCalibrationPointClick(index, element) {
    if (index !== calibrationIndex) return;
    const rect = element.getBoundingClientRect();
    if (window.webgazer && typeof window.webgazer.recordScreenPosition === 'function') {
      window.webgazer.recordScreenPosition(rect.left + rect.width / 2, rect.top + rect.height / 2, 'click');
    }
    calibrationClicks += 1;
    if (calibrationClicks < CALIBRATION_CLICKS_PER_POINT) {
      element.textContent = `${calibrationClicks}/${CALIBRATION_CLICKS_PER_POINT}`;
      element.style.boxShadow = `0 0 0 ${5 + calibrationClicks * 2}px rgba(120,242,203,.13)`;
      updateCalibrationProgress();
      return;
    }
    element.classList.add('done');
    element.textContent = '✓';
    calibrationIndex += 1;
    calibrationClicks = 0;
    if (calibrationIndex >= CALIBRATION_POSITIONS.length) {
      calibrationProgress.style.width = '100%';
      hide(calibrationOverlay);
      startGame(webcamGameMode);
      return;
    }
    updateCalibrationProgress();
  }

  function newPhaseStat() { return { samples: 0, hits: 0 }; }

  function scoreFor(stats) {
    return stats && stats.samples ? Math.round((stats.hits / stats.samples) * 100) : null;
  }

  function currentPhaseIndex(elapsedMs) {
    return Math.max(0, Math.min(2, Math.floor(elapsedMs / (PHASE_SECONDS * 1000))));
  }

  function recordSample(stats, distance, threshold) {
    stats.samples += 1;
    if (distance <= threshold) stats.hits += 1;
  }

  function startGame(nextMode) {
    mode = nextMode;
    running = true;
    remainingSeconds = TEST_SECONDS;
    totalSamples = 0;
    hitSamples = 0;
    mouseTotalSamples = 0;
    mouseHitSamples = 0;
    phaseStats = [newPhaseStat(), newPhaseStat(), newPhaseStat()];
    mousePhaseStats = [newPhaseStat(), newPhaseStat(), newPhaseStat()];
    latestPoint = null;
    latestPointTime = 0;
    latestMousePoint = null;
    latestMousePointTime = 0;
    mouseInside = false;
    sessionStartedMs = Date.now();
    currentSessionId = createSessionId();
    gazeBuffer = [];
    filteredGaze = null;
    touchActive = false;
    pointerInside = false;
    updateAccuracy();
    timerText.textContent = formatTime(remainingSeconds);
    hide(startOverlay); hide(calibrationOverlay); hide(resultOverlay);
    playfield.classList.add('playing');
    centerInstruction.style.opacity = '0';
    reticle.classList.remove('visible');
    if (nextMode === 'webcam') {
      setStatus('공을 눈으로 따라가 보세요', true);
      modePill.textContent = '웹캠 시선 추적';
    } else if (nextMode === 'hybrid') {
      setStatus('공을 눈으로 따라가고 마우스도 움직여 보세요', true);
      modePill.textContent = '시선 + 마우스';
    } else {
      setStatus('공을 터치 또는 마우스로 따라가 보세요', true);
      modePill.textContent = '터치·마우스';
    }
    startTargetAnimation();
    clearInterval(sampleTimer);
    clearInterval(gameClock);
    sampleTimer = window.setInterval(samplePoint, SAMPLE_INTERVAL_MS);
    gameClock = window.setInterval(() => {
      remainingSeconds -= 1;
      timerText.textContent = formatTime(remainingSeconds);
      if (remainingSeconds <= 0) finishGame();
    }, 1000);
  }

  function samplePoint() {
    if (!running) return;
    const now = Date.now();
    const elapsedMs = Math.max(0, now - sessionStartedMs);
    const phaseIndex = currentPhaseIndex(elapsedMs);
    const targetRect = target.getBoundingClientRect();
    const targetX = targetRect.left + targetRect.width / 2;
    const targetY = targetRect.top + targetRect.height / 2;
    const stageWidth = playfield.getBoundingClientRect().width;
    const threshold = Math.max(42, Math.min(82, stageWidth * 0.075));

    // Webcam and hybrid use gaze as the primary metric.
    if (mode === 'webcam' || mode === 'hybrid') {
      if (latestPoint && now - latestPointTime <= 550) {
        const gazeDistance = Math.hypot(latestPoint.x - targetX, latestPoint.y - targetY);
        recordSample(phaseStats[phaseIndex], gazeDistance, threshold);
        totalSamples += 1;
        if (gazeDistance <= threshold) hitSamples += 1;
        updateAccuracy();
      }
    } else if (mode === 'touch') {
      const pointerReady = latestPoint && pointerInside
        && (latestPoint.pointerType === 'mouse' || touchActive);
      if (pointerReady) {
        const pointerDistance = Math.hypot(latestPoint.x - targetX, latestPoint.y - targetY);
        recordSample(phaseStats[phaseIndex], pointerDistance, threshold);
        totalSamples += 1;
        if (pointerDistance <= threshold) hitSamples += 1;
        updateAccuracy();
      }
    }

    // Hybrid mode also samples the physical mouse pointer separately, for comparison only.
    if (mode === 'hybrid' && latestMousePoint && mouseInside) {
      const mouseDistance = Math.hypot(latestMousePoint.x - targetX, latestMousePoint.y - targetY);
      recordSample(mousePhaseStats[phaseIndex], mouseDistance, threshold);
      mouseTotalSamples += 1;
      if (mouseDistance <= threshold) mouseHitSamples += 1;
    }
  }

  function createSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') return window.crypto.randomUUID();
    return `s-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }

  function setSaveStatus(message, error = false) {
    if (!saveStatus) return;
    saveStatus.textContent = message;
    saveStatus.classList.toggle('error', error);
  }

  async function saveResultToAdmin(payload) {
    if (!ADMIN_SHEETS_WEB_APP_URL || !/^https:\/\/script\.google\.com\/macros\/s\/[^\s]+\/exec(?:$|\?)/.test(ADMIN_SHEETS_WEB_APP_URL)) {
      setSaveStatus('관리자 저장 URL이 아직 설정되지 않았어요. README의 Google Sheets 연결 단계를 따라 설정해 주세요.');
      return;
    }
    setSaveStatus('관리자 기록으로 전송 중…');
    try {
      // text/plain + no-cors는 브라우저의 CORS preflight를 피합니다. 응답 본문을 읽을 수 없으므로 시트에서 저장 여부를 확인해야 해요.
      await fetch(ADMIN_SHEETS_WEB_APP_URL, {
        method: 'POST',
        mode: 'no-cors',
        headers: { 'Content-Type': 'text/plain;charset=UTF-8' },
        body: JSON.stringify(payload),
        keepalive: true
      });
      setSaveStatus('저장 요청을 보냈어요. 관리자 Google Sheets에 새 행이 생겼는지 확인해 주세요.');
    } catch (error) {
      console.error('관리자 시트 저장 요청 실패:', error);
      setSaveStatus('전송에 실패했어요. 인터넷 연결과 Apps Script 배포 URL을 확인해 주세요.', true);
    }
  }

  function updatePhaseResult(prefix, stat) {
    const score = scoreFor(stat);
    $(`${prefix}Score`).textContent = score == null ? '--%' : `${score}%`;
    $(`${prefix}Samples`).textContent = `측정 ${stat.samples}회`;
  }

  function renderPhaseResults() {
    $('phaseHeading').textContent = mode === 'hybrid' ? '초반 · 중반 · 후반 시선 지표' : mode === 'webcam' ? '초반 · 중반 · 후반 시선 지표' : '초반 · 중반 · 후반 포인터 지표';
    updatePhaseResult('phaseEarly', phaseStats[0]);
    updatePhaseResult('phaseMiddle', phaseStats[1]);
    updatePhaseResult('phaseLate', phaseStats[2]);
    const mouseWrap = $('mousePhaseWrap');
    if (mode === 'hybrid') {
      updatePhaseResult('mousePhaseEarly', mousePhaseStats[0]);
      updatePhaseResult('mousePhaseMiddle', mousePhaseStats[1]);
      updatePhaseResult('mousePhaseLate', mousePhaseStats[2]);
      show(mouseWrap);
    } else {
      hide(mouseWrap);
    }
  }

  function phasePayload(stats) {
    return {
      earlyScorePercent: scoreFor(stats[0]),
      earlySamples: stats[0].samples,
      earlyHits: stats[0].hits,
      middleScorePercent: scoreFor(stats[1]),
      middleSamples: stats[1].samples,
      middleHits: stats[1].hits,
      lateScorePercent: scoreFor(stats[2]),
      lateSamples: stats[2].samples,
      lateHits: stats[2].hits
    };
  }

  function finishGame() {
    running = false;
    cancelAnimationFrame(animationFrame);
    clearInterval(sampleTimer);
    clearInterval(gameClock);
    touchActive = false;
    pointerInside = false;
    reticle.classList.remove('visible');
    setStatus('테스트 완료', false);
    modePill.textContent = '완료';
    const percent = totalSamples ? Math.round(hitSamples / totalSamples * 100) : 0;
    $('resultAccuracy').textContent = totalSamples ? `${percent}%` : '측정 없음';
    $('resultSamples').textContent = String(totalSamples);
    $('resultHits').textContent = String(hitSamples);
    $('resultMode').textContent = mode === 'webcam' ? '시선' : mode === 'hybrid' ? '시선+마우스' : '터치/마우스';
    renderPhaseResults();
    $('resultMessage').textContent = totalSamples
      ? (mode === 'hybrid'
        ? '전체 점수는 웹캠 시선 추정 기준이며, 아래에서 마우스 포인터 기록과 구간별로 비교할 수 있어요.'
        : percent >= 65 ? '측정된 시선 또는 포인터가 목표 가까이 있었던 비율이 높았어요.' : percent >= 35 ? '천천히 움직이는 공을 따라가며 한 번 더 해 보세요.' : '화면 거리와 조명을 확인하고 다시 시도해 보세요.')
      : '측정 데이터가 충분하지 않았어요. 권한과 보정 상태를 확인하거나 터치 모드로 다시 시도해 주세요.';
    show(resultOverlay);

    const payload = {
      app: 'FOCUS_LAB',
      sessionId: currentSessionId || createSessionId(),
      participantCode: (participantCodeInput?.value || '').trim().slice(0, 32),
      mode,
      measurementType: mode === 'webcam' ? 'gaze-estimate' : mode === 'hybrid' ? 'gaze-and-pointer' : 'pointer-demo',
      scorePercent: percent,
      totalSamples,
      hitSamples,
      phaseScores: phasePayload(phaseStats),
      mouseScorePercent: mouseTotalSamples ? Math.round((mouseHitSamples / mouseTotalSamples) * 100) : null,
      mouseTotalSamples,
      mouseHitSamples,
      mousePhaseScores: phasePayload(mousePhaseStats),
      testSeconds: TEST_SECONDS,
      actualElapsedSeconds: sessionStartedMs ? Math.min(TEST_SECONDS, Math.round((Date.now() - sessionStartedMs) / 1000)) : TEST_SECONDS,
      paceSeconds: TARGET_MOVE_SECONDS,
      deviceType: window.matchMedia && window.matchMedia('(pointer: coarse)').matches ? 'touch-device' : 'mouse-device',
      filterVersion: mode === 'webcam' || mode === 'hybrid' ? GAZE_FILTER_VERSION : 'not-applicable'
    };
    saveResultToAdmin(payload);
  }

  function resetToStart() {
    running = false;
    cancelAnimationFrame(animationFrame);
    clearInterval(sampleTimer);
    clearInterval(gameClock);
    latestPoint = null;
    latestPointTime = 0;
    latestMousePoint = null;
    latestMousePointTime = 0;
    mouseInside = false;
    gazeBuffer = [];
    filteredGaze = null;
    pointerInside = false;
    touchActive = false;
    hide(resultOverlay); hide(calibrationOverlay); show(startOverlay);
    timerText.textContent = formatTime(TEST_SECONDS);
    accuracyText.innerHTML = `--<small>%</small>`;
    accuracyBar.style.width = '0%';
    accuracyTrack.setAttribute('aria-valuenow', '0');
    reticle.classList.remove('visible');
    playfield.classList.remove('playing');
    centerInstruction.style.opacity = '';
    setStatus('준비 완료', false);
    webcamStartButton.disabled = false;
    hybridStartButton.disabled = false;
    webcamStartButton.innerHTML = '<span class="button-icon">◎</span> 웹캠 시선 추적으로 시작 <span class="button-arrow">→</span>';
  }

  function onPointerMove(event) {
    if (!running || (mode !== 'touch' && mode !== 'hybrid')) return;
    const rect = playfield.getBoundingClientRect();
    const x = event.clientX;
    const y = event.clientY;
    mouseInside = x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
    if (mode === 'touch') pointerInside = mouseInside;
    if (mouseInside) {
      lastPointerType = event.pointerType || lastPointerType;
      latestMousePoint = { x, y, pointerType: lastPointerType };
      latestMousePointTime = Date.now();
      if (mode === 'touch') updateLatestPoint(x, y, 'touch', lastPointerType);
    }
  }
  playfield.addEventListener('pointerdown', (event) => {
    if (running && (mode === 'touch' || mode === 'hybrid')) {
      lastPointerType = event.pointerType || 'mouse';
      touchActive = lastPointerType !== 'mouse';
      mouseInside = true;
      if (mode === 'touch') pointerInside = true;
      latestMousePoint = { x: event.clientX, y: event.clientY, pointerType: lastPointerType };
      latestMousePointTime = Date.now();
      if (mode === 'touch') updateLatestPoint(event.clientX, event.clientY, 'touch', lastPointerType);
    }
  });
  playfield.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', () => {
    if (mode === 'touch' || mode === 'hybrid') touchActive = false;
  });
  playfield.addEventListener('pointerleave', () => {
    if (mode === 'touch' || mode === 'hybrid') mouseInside = false;
    if (mode === 'touch') pointerInside = false;
  });

  webcamStartButton.addEventListener('click', () => startWebcamFlow('webcam'));
  hybridStartButton.addEventListener('click', () => startWebcamFlow('hybrid'));
  touchStartButton.addEventListener('click', () => startGame('touch'));
  restartButton.addEventListener('click', resetToStart);
  cancelCalibrationButton.addEventListener('click', () => {
    hide(calibrationOverlay); show(startOverlay);
    webcamStartButton.disabled = false;
    hybridStartButton.disabled = false;
    setStatus('보정 취소됨', false);
  });

  if (window.webgazer) {
    setLibraryMessage('라이브러리 준비 완료 · 시선 추적, 시선+마우스 비교, 터치 모드 중 선택하세요.');
  } else {
    window.addEventListener('load', () => {
      if (window.webgazer) setLibraryMessage('라이브러리 준비 완료 · 시선 추적, 시선+마우스 비교, 터치 모드 중 선택하세요.');
      else setLibraryMessage('라이브러리를 불러오지 못했어요. 인터넷 연결을 확인하고 터치 모드로 먼저 시험해 보세요.', true);
    }, { once: true });
  }
})();
