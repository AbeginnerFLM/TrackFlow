import { $, clamp } from '../shared/dom.js'
import { perspectiveTransform, undistortPoint, groundToLatLon, latLonToLocal } from '../shared/geo.js'
import { downloadText } from '../shared/download.js'

const vid = $('vid')
const cvs = $('cvs')
const cx = cvs.getContext('2d')

let currentFrame = 0
let totalFrames = 0
let videoLoaded = false
let mode = 'gcp'
let cameraMatrix = null
let distCoeffs = null
let H = null
let gcpPoints = []
let valResults = []
let valSelected = new Set()
let pendingClick = null
let sourceVideoUrl = null

function drawFrame() {
  if (!videoLoaded) return
  cx.drawImage(vid, 0, 0, cvs.width, cvs.height)
  for (let i = 0; i < gcpPoints.length; i++) {
    const p = gcpPoints[i]
    cx.beginPath(); cx.arc(p.imgX, p.imgY, 6, 0, Math.PI * 2)
    cx.fillStyle = 'rgba(72,240,139,.7)'; cx.fill()
    cx.strokeStyle = '#fff'; cx.lineWidth = 2; cx.stroke()
    cx.fillStyle = '#fff'; cx.font = 'bold 11px sans-serif'; cx.textAlign = 'center'
    cx.fillText(String(i + 1), p.imgX, p.imgY - 10)
  }
  for (let i = 0; i < valResults.length; i++) {
    const r = valResults[i]
    cx.beginPath(); cx.arc(r.imgX, r.imgY, 5, 0, Math.PI * 2)
    cx.fillStyle = valSelected.has(i) ? '#f28b82' : 'rgba(148,103,255,.7)'; cx.fill()
    cx.strokeStyle = '#fff'; cx.lineWidth = 1.5; cx.stroke()
  }
}

function seekFrame(idx) {
  if (!videoLoaded) return
  idx = clamp(Math.round(idx), 0, Math.max(0, totalFrames - 1))
  currentFrame = idx
  $('slider').value = idx
  $('frameNum').value = idx
  vid.currentTime = idx / 30
  vid.addEventListener('seeked', function handler() {
    vid.removeEventListener('seeked', handler)
    drawFrame()
  })
}

function loadVideo(file) {
  if (!file) return
  if (sourceVideoUrl) URL.revokeObjectURL(sourceVideoUrl)
  sourceVideoUrl = URL.createObjectURL(file)
  vid.src = sourceVideoUrl
  vid.onloadedmetadata = () => {
    totalFrames = Math.round(vid.duration * 30)
    $('slider').max = Math.max(0, totalFrames - 1)
    $('totalFrames').value = totalFrames
    videoLoaded = true
    cvs.width = vid.videoWidth
    cvs.height = vid.videoHeight
    seekFrame(0)
    $('frameInfo').textContent = `${vid.videoWidth}x${vid.videoHeight} ~${totalFrames} frames`
  }
}

function updateGcpUI() {
  $('gcpList').innerHTML = gcpPoints.map((p, i) => `<div class="gcp-item"><span>${i + 1}: px(${p.imgX},${p.imgY}) → (${p.lon.toFixed(7)}, ${p.lat.toFixed(7)})</span><span class="del" data-del="${i}">✕</span></div>`).join('')
  $('gcpStatus').textContent = `${gcpPoints.length} points${gcpPoints.length < 4 ? ' — need at least 4' : ''}`
  $('gcpStatus').className = 'status' + (gcpPoints.length >= 4 ? ' ok' : '')
}

function updateValUI() {
  $('valList').innerHTML = valResults.map((r, i) => `<div class="val-item" data-select="${i}"><span>V${i + 1} f${r.frame} err=<b>${r.errorM.toFixed(3)}m</b> px(${r.imgX},${r.imgY})</span></div>`).join('')
  if (valResults.length > 0) {
    const rmse = Math.sqrt(valResults.reduce((s, r) => s + r.errorM ** 2, 0) / valResults.length)
    $('valStatus').textContent = `Total RMSE: ${rmse.toFixed(4)} m (${valResults.length} points)`
    $('valStatus').className = 'status ok'
  }
}

function solveDLT4(src, dst) {
  const n = src.length
  if (n < 4) return null
  const A = []
  for (let i = 0; i < n; i++) {
    const [x, y] = src[i]
    const [u, v] = dst[i]
    A.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
    A.push([0, 0, 0, -x, -y, -1, v * x, v * y, v])
  }
  const cols = 9
  const AtA = Array.from({ length: cols }, () => new Float64Array(cols))
  for (let i = 0; i < cols; i++) {
    for (let j = i; j < cols; j++) {
      let sum = 0
      for (let k = 0; k < A.length; k++) sum += A[k][i] * A[k][j]
      AtA[i][j] = sum
      AtA[j][i] = sum
    }
  }
  let shift = 0
  for (let i = 0; i < cols; i++) {
    let r = 0
    for (let j = 0; j < cols; j++) if (j !== i) r += Math.abs(AtA[i][j])
    shift = Math.max(shift, AtA[i][i] + r)
  }
  shift *= 1.01
  const M = Array.from({ length: cols }, (_, i) => {
    const row = new Float64Array(cols)
    for (let j = 0; j < cols; j++) row[j] = -AtA[i][j]
    row[i] += shift
    return row
  })
  const LU = M.map((row) => Float64Array.from(row))
  const piv = Array.from({ length: cols }, (_, i) => i)
  for (let k = 0; k < cols; k++) {
    let mx = Math.abs(LU[k][k]), mi = k
    for (let i = k + 1; i < cols; i++) {
      const v = Math.abs(LU[i][k])
      if (v > mx) { mx = v; mi = i }
    }
    if (mi !== k) { [LU[k], LU[mi]] = [LU[mi], LU[k]]; [piv[k], piv[mi]] = [piv[mi], piv[k]] }
    if (Math.abs(LU[k][k]) < 1e-14) return null
    for (let i = k + 1; i < cols; i++) {
      LU[i][k] /= LU[k][k]
      for (let j = k + 1; j < cols; j++) LU[i][j] -= LU[i][k] * LU[k][j]
    }
  }
  function luSolve(b) {
    const x = new Float64Array(cols)
    const pb = new Float64Array(cols)
    for (let i = 0; i < cols; i++) pb[i] = b[piv[i]]
    for (let i = 0; i < cols; i++) {
      x[i] = pb[i]
      for (let j = 0; j < i; j++) x[i] -= LU[i][j] * x[j]
    }
    for (let i = cols - 1; i >= 0; i--) {
      for (let j = i + 1; j < cols; j++) x[i] -= LU[i][j] * x[j]
      x[i] /= LU[i][i]
    }
    return x
  }
  let v = new Float64Array(cols).fill(1)
  for (let iter = 0; iter < 30; iter++) {
    v = luSolve(v)
    let norm = 0
    for (let i = 0; i < cols; i++) norm += v[i] * v[i]
    norm = Math.sqrt(norm)
    if (norm < 1e-30) return null
    for (let i = 0; i < cols; i++) v[i] /= norm
  }
  if (Math.abs(v[8]) > 1e-12) for (let i = 0; i < 9; i++) v[i] /= v[8]
  return Array.from(v)
}

function reprojectError(Hm, src, dst) {
  const gx = Hm[0] * src[0] + Hm[1] * src[1] + Hm[2]
  const gy = Hm[3] * src[0] + Hm[4] * src[1] + Hm[5]
  const w = Hm[6] * src[0] + Hm[7] * src[1] + Hm[8]
  if (Math.abs(w) < 1e-12) return 1e10
  const dx = gx / w - dst[0]
  const dy = gy / w - dst[1]
  return Math.sqrt(dx * dx + dy * dy)
}

function findHomographyRANSAC(srcPts, dstPts, thresh, maxIter) {
  const n = srcPts.length
  if (n < 4) return null
  if (n === 4) return { H: solveDLT4(srcPts, dstPts), inliers: new Array(4).fill(true), rmse: 0 }
  let bestH = null, bestInliers = [], bestCount = 0
  for (let iter = 0; iter < maxIter; iter++) {
    const idx = new Set()
    while (idx.size < 4) idx.add(Math.floor(Math.random() * n))
    const s = [], d = []
    for (const i of idx) { s.push(srcPts[i]); d.push(dstPts[i]) }
    const candidate = solveDLT4(s, d)
    if (!candidate) continue
    const inliers = []
    let count = 0
    for (let i = 0; i < n; i++) {
      const ok = reprojectError(candidate, srcPts[i], dstPts[i]) < thresh
      inliers.push(ok)
      if (ok) count++
    }
    if (count > bestCount) {
      bestCount = count
      bestH = candidate
      bestInliers = inliers
    }
  }
  if (!bestH) return null
  let sse = 0, cnt = 0
  for (let i = 0; i < n; i++) {
    if (bestInliers[i]) {
      const err = reprojectError(bestH, srcPts[i], dstPts[i])
      sse += err * err
      cnt++
    }
  }
  return { H: bestH, inliers: bestInliers, rmse: cnt ? Math.sqrt(sse / cnt) : NaN }
}

function formatH(Hm) {
  return [0, 1, 2].map((r) => [0, 1, 2].map((c) => Hm[r * 3 + c].toExponential(8)).join('  ')).join('\n')
}

function computeH() {
  if (gcpPoints.length < 4) return alert('Need at least 4 GCP points')
  const originLon = +$('oriLon').value
  const originLat = +$('oriLat').value
  if (!originLon && !originLat) return alert('Set origin lon/lat first')
  const doUndist = +$('undistGcp').value && cameraMatrix && distCoeffs
  const src = []
  const dst = []
  for (const p of gcpPoints) {
    let px = p.imgX, py = p.imgY
    if (doUndist) [px, py] = undistortPoint(px, py, cameraMatrix, distCoeffs)
    src.push([px, py])
    dst.push(latLonToLocal(p.lon, p.lat, originLon, originLat))
  }
  const result = findHomographyRANSAC(src, dst, +$('ransacThresh').value || 3, 1000)
  if (!result?.H) return
  H = result.H
  $('hStatus').textContent = `RMSE: ${result.rmse.toFixed(4)} m | Inliers: ${result.inliers.filter(Boolean).length}/${gcpPoints.length}`
  $('hStatus').className = 'status ok'
  $('hMatrix').style.display = 'block'
  $('hMatrix').textContent = formatH(H)
  drawFrame()
}

function loadCamParams(file) {
  if (!file) return
  const reader = new FileReader()
  reader.onload = (event) => {
    try {
      const data = JSON.parse(event.target.result)
      const K = data.camera_matrix || data.K
      const dist = data.dist_coeffs || data.dist
      if (!K || !dist) throw new Error('Missing camera_matrix/K or dist_coeffs/dist')
      cameraMatrix = Array.isArray(K[0]) ? K.flat() : Array.from(K)
      distCoeffs = Array.isArray(dist[0]) ? dist.flat() : Array.from(dist)
      $('camStatus').textContent = `Loaded: K(${cameraMatrix.length}) dist(${distCoeffs.length})`
      $('camStatus').className = 'status ok'
    } catch (error) {
      $('camStatus').textContent = `Error: ${error.message}`
      $('camStatus').className = 'status err'
    }
  }
  reader.readAsText(file)
}

function applyH() {
  const matches = $('hInput').value.match(/[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?/g)
  if (!matches || matches.length !== 9) return alert('Need exactly 9 numbers')
  H = matches.map(Number)
  if (Math.abs(H[8]) > 1e-12) H = H.map((v) => v / H[8])
  $('hStatus').textContent = 'H applied (loaded)'
  $('hStatus').className = 'status ok'
  $('hMatrix').style.display = 'block'
  $('hMatrix').textContent = formatH(H)
  drawFrame()
}

function confirmGcp() {
  const txt = $('gcpLonLat').value.trim()
  if (!txt) return cancelGcp()
  const parts = txt.split(/[\s,]+/)
  if (parts.length !== 2) return alert('Enter lon lat separated by space')
  const lon = parseFloat(parts[0]), lat = parseFloat(parts[1])
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return alert('Invalid numbers')
  gcpPoints.push({ imgX: pendingClick.x, imgY: pendingClick.y, lon, lat })
  if (gcpPoints.length === 1 && !+$('oriLon').value && !+$('oriLat').value) {
    $('oriLon').value = lon
    $('oriLat').value = lat
  }
  $('gcpModal').style.display = 'none'
  updateGcpUI(); drawFrame()
}
function cancelGcp() { $('gcpModal').style.display = 'none' }

function confirmVal() {
  const imgX = pendingClick.x, imgY = pendingClick.y
  let ux = imgX, uy = imgY
  if (cameraMatrix && distCoeffs) [ux, uy] = undistortPoint(imgX, imgY, cameraMatrix, distCoeffs)
  const gp = perspectiveTransform(ux, uy, H)
  const predGx = gp[0], predGy = gp[1]
  const ll = groundToLatLon(predGx, predGy, +$('oriLon').value, +$('oriLat').value)
  let trueLon = ll ? ll[1] : null, trueLat = ll ? ll[0] : null, trueGx = predGx, trueGy = predGy
  const txt = $('valTrue').value.trim()
  if (txt) {
    const parts = txt.split(/[\s,]+/)
    if (parts.length >= 2) {
      trueLon = parseFloat(parts[0]); trueLat = parseFloat(parts[1])
      ;[trueGx, trueGy] = latLonToLocal(trueLon, trueLat, +$('oriLon').value, +$('oriLat').value)
    }
  }
  const errorM = Math.sqrt((predGx - trueGx) ** 2 + (predGy - trueGy) ** 2)
  valResults.push({ frame: currentFrame, imgX, imgY, predGx, predGy, predLon: ll ? ll[1] : null, predLat: ll ? ll[0] : null, trueGx, trueGy, trueLon, trueLat, errorM })
  $('valModal').style.display = 'none'
  updateValUI(); drawFrame()
}
function cancelVal() { $('valModal').style.display = 'none' }

function measureDist() {
  if (valSelected.size !== 2) return alert('Select exactly 2 validation points')
  const [i, j] = [...valSelected]
  const r1 = valResults[i], r2 = valResults[j]
  const dPred = Math.sqrt((r1.predGx - r2.predGx) ** 2 + (r1.predGy - r2.predGy) ** 2)
  const dTrue = Math.sqrt((r1.trueGx - r2.trueGx) ** 2 + (r1.trueGy - r2.trueGy) ** 2)
  alert(`Predicted distance: ${dPred.toFixed(3)} m\nTrue distance: ${dTrue.toFixed(3)} m\nDifference: ${(dPred - dTrue).toFixed(3)} m`)
}

function exportValCsv() {
  let csv = 'frame,img_x,img_y,pred_gx,pred_gy,pred_lon,pred_lat,true_gx,true_gy,true_lon,true_lat,error_m\n'
  for (const r of valResults) {
    csv += [r.frame, r.imgX, r.imgY, r.predGx.toFixed(6), r.predGy.toFixed(6), r.predLon ?? '', r.predLat ?? '', r.trueGx.toFixed(6), r.trueGy.toFixed(6), r.trueLon ?? '', r.trueLat ?? '', r.errorM.toFixed(6)].join(',') + '\n'
  }
  downloadText(csv, 'validation_results.csv')
}

function exportGcpCsv() {
  let csv = 'image_x,image_y,lon,lat\n'
  for (const p of gcpPoints) csv += `${p.imgX},${p.imgY},${p.lon},${p.lat}\n`
  downloadText(csv, 'gcp_points.csv')
}

function loadHFile(file) {
  if (!file) return
  const reader = new FileReader()
  reader.onload = (event) => { $('hInput').value = event.target.result }
  reader.readAsText(file)
}

function setMode(nextMode) {
  mode = nextMode
  $('modeGcp').classList.toggle('active', mode === 'gcp')
  $('modeVal').classList.toggle('active', mode === 'validate')
}

function bindEvents() {
  $('openVideoBtn').onclick = () => $('vidFile').click()
  $('vidFile').onchange = (e) => loadVideo(e.target.files[0])
  $('slider').oninput = (e) => seekFrame(+e.target.value)
  $('frameNum').onchange = (e) => seekFrame(+e.target.value)
  $('prevFrameBtn').onclick = () => seekFrame(currentFrame - 1)
  $('nextFrameBtn').onclick = () => seekFrame(currentFrame + 1)
  $('openCamBtn').onclick = () => $('camFile').click()
  $('camFile').onchange = (e) => loadCamParams(e.target.files[0])
  $('importGcpBtn').onclick = () => $('gcpFile').click()
  $('gcpFile').onchange = (event) => {
    const file = event.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (loadEvent) => {
      const lines = String(loadEvent.target.result || '').trim().split('\n')
      if (lines.length < 2) return
      const header = lines[0].toLowerCase().split(',').map((s) => s.trim())
      const iX = header.indexOf('image_x') >= 0 ? header.indexOf('image_x') : header.indexOf('u')
      const iY = header.indexOf('image_y') >= 0 ? header.indexOf('image_y') : header.indexOf('v')
      const iLon = header.indexOf('lon') >= 0 ? header.indexOf('lon') : header.indexOf('longitude')
      const iLat = header.indexOf('lat') >= 0 ? header.indexOf('lat') : header.indexOf('latitude')
      if (iX < 0 || iY < 0) return
      gcpPoints = []
      for (let r = 1; r < lines.length; r++) {
        const cols = lines[r].split(',').map((s) => s.trim())
        if (cols.length < 2) continue
        gcpPoints.push({ imgX: +cols[iX], imgY: +cols[iY], lon: iLon >= 0 ? +cols[iLon] : 0, lat: iLat >= 0 ? +cols[iLat] : 0 })
      }
      updateGcpUI(); drawFrame()
    }
    reader.readAsText(file)
  }
  $('exportGcpBtn').onclick = exportGcpCsv
  $('clearGcpBtn').onclick = () => { gcpPoints = []; updateGcpUI(); drawFrame() }
  $('computeHBtn').onclick = computeH
  $('copyHBtn').onclick = () => { if (H) navigator.clipboard.writeText(H.join(' ')) }
  $('exportHBtn').onclick = () => { if (H) downloadText(formatH(H), 'homography.txt') }
  $('measureBtn').onclick = measureDist
  $('exportValBtn').onclick = exportValCsv
  $('clearValBtn').onclick = () => { valResults = []; valSelected.clear(); updateValUI(); drawFrame() }
  $('openHBtn').onclick = () => $('hFile').click()
  $('hFile').onchange = (e) => loadHFile(e.target.files[0])
  $('applyHBtn').onclick = applyH
  $('modeGcp').onclick = () => setMode('gcp')
  $('modeVal').onclick = () => setMode('validate')
  $('confirmGcpBtn').onclick = confirmGcp
  $('cancelGcpBtn').onclick = cancelGcp
  $('confirmValBtn').onclick = confirmVal
  $('cancelValBtn').onclick = cancelVal

  $('gcpList').onclick = (event) => {
    const target = event.target
    if (target instanceof HTMLElement && target.dataset.del) {
      gcpPoints.splice(+target.dataset.del, 1)
      updateGcpUI(); drawFrame()
    }
  }
  $('valList').onclick = (event) => {
    const target = event.target.closest('[data-select]')
    if (target) {
      const index = +target.dataset.select
      if (valSelected.has(index)) valSelected.delete(index)
      else {
        if (valSelected.size >= 2) valSelected.clear()
        valSelected.add(index)
      }
      updateValUI(); drawFrame()
    }
  }

  cvs.addEventListener('click', (event) => {
    if (!videoLoaded) return
    const rect = cvs.getBoundingClientRect()
    const sx = cvs.width / rect.width
    const sy = cvs.height / rect.height
    const imgX = Math.round((event.clientX - rect.left) * sx)
    const imgY = Math.round((event.clientY - rect.top) * sy)
    if (mode === 'gcp') {
      pendingClick = { x: imgX, y: imgY }
      $('gcpPixel').textContent = `(${imgX}, ${imgY})`
      $('gcpLonLat').value = ''
      $('gcpModal').style.display = 'flex'
    } else {
      if (!H) return alert('Please compute or load a Homography first.')
      pendingClick = { x: imgX, y: imgY }
      $('valPixel').textContent = `(${imgX}, ${imgY})`
      let ux = imgX, uy = imgY
      if (cameraMatrix && distCoeffs) [ux, uy] = undistortPoint(imgX, imgY, cameraMatrix, distCoeffs)
      const gp = perspectiveTransform(ux, uy, H)
      const ll = groundToLatLon(gp[0], gp[1], +$('oriLon').value, +$('oriLat').value)
      $('valPred').textContent = `Predicted ground: (${gp[0].toFixed(4)}, ${gp[1].toFixed(4)})${ll ? `\nPredicted lon/lat: (${ll[1].toFixed(9)}, ${ll[0].toFixed(9)})` : ''}`
      $('valTrue').value = ll ? `${ll[1].toFixed(9)} ${ll[0].toFixed(9)}` : ''
      $('valModal').style.display = 'flex'
    }
  })

  cvs.addEventListener('mousemove', (event) => {
    if (!videoLoaded) return
    const rect = cvs.getBoundingClientRect()
    const sx = cvs.width / rect.width
    const sy = cvs.height / rect.height
    const ix = Math.round((event.clientX - rect.left) * sx)
    const iy = Math.round((event.clientY - rect.top) * sy)
    let info = `px(${ix},${iy})`
    if (H) {
      let ux = ix, uy = iy
      if (cameraMatrix && distCoeffs) [ux, uy] = undistortPoint(ix, iy, cameraMatrix, distCoeffs)
      const gp = perspectiveTransform(ux, uy, H)
      if (gp) info += ` → ground(${gp[0].toFixed(2)},${gp[1].toFixed(2)})`
    }
    $('cursorInfo').textContent = info
  })
}

setMode('gcp')
bindEvents()
