export function perspectiveTransform(px, py, H) {
  if (!H) return null
  const w = H[6] * px + H[7] * py + H[8]
  if (Math.abs(w) < 1e-12) return null
  return [
    (H[0] * px + H[1] * py + H[2]) / w,
    (H[3] * px + H[4] * py + H[5]) / w,
  ]
}

export function undistortPoint(px, py, K, dist) {
  if (!K || !dist) return [px, py]
  const fx = K[0], fy = K[4], cx = K[2], cy = K[5]
  let x = (px - cx) / fx
  let y = (py - cy) / fy
  const k1 = dist[0] || 0
  const k2 = dist[1] || 0
  const p1 = dist[2] || 0
  const p2 = dist[3] || 0
  const k3 = dist[4] || 0
  const x0 = x, y0 = y
  for (let i = 0; i < 10; i++) {
    const r2 = x * x + y * y
    const r4 = r2 * r2
    const r6 = r4 * r2
    const radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    const dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    const dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    x = (x0 - dx) / radial
    y = (y0 - dy) / radial
  }
  return [x * fx + cx, y * fy + cy]
}

export function latLonToUtm(lat, lon) {
  const zone = Math.floor((lon + 180) / 6) + 1
  const isNorth = lat >= 0
  const k0 = 0.9996
  const a = 6378137
  const f = 1 / 298.257223563
  const n = f / (2 - f)
  const n2 = n * n
  const n3 = n2 * n
  const n4 = n3 * n
  const A = a / (1 + n) * (1 + n2 / 4 + n4 / 64)
  const latR = lat * Math.PI / 180
  const lonR = lon * Math.PI / 180
  const lon0R = (zone * 6 - 183) * Math.PI / 180
  const t = Math.sinh(Math.atanh(Math.sin(latR)) - 2 * n / (1 + n) * Math.atanh(2 * n / (1 + n) * Math.sin(latR)))
  const xi0 = Math.atan2(t, Math.cos(lonR - lon0R))
  const eta0 = Math.atanh(Math.sin(lonR - lon0R) / Math.sqrt(1 + t * t))
  const a1 = 0.5 * n - 2 / 3 * n2 + 5 / 16 * n3
  const a2 = 13 / 48 * n2 - 3 / 5 * n3
  const a3 = 61 / 240 * n3
  let xi = xi0
  let eta = eta0
  for (let j = 1; j <= 3; j++) {
    const aj = [0, a1, a2, a3][j]
    xi += aj * Math.sin(2 * j * xi0) * Math.cosh(2 * j * eta0)
    eta += aj * Math.cos(2 * j * xi0) * Math.sinh(2 * j * eta0)
  }
  const easting = k0 * A * eta + 500000
  const northing = k0 * A * xi + (isNorth ? 0 : 10000000)
  return [easting, northing, zone, isNorth]
}

export function utmToLatLon(easting, northing, zone, isNorth) {
  const k0 = 0.9996
  const a = 6378137
  const f = 1 / 298.257223563
  const n = f / (2 - f)
  const n2 = n * n
  const n3 = n2 * n
  const n4 = n3 * n
  const A = a / (1 + n) * (1 + n2 / 4 + n4 / 64)
  const x = easting - 500000
  const y = isNorth ? northing : northing - 10000000
  const xi = y / (k0 * A)
  const eta = x / (k0 * A)
  const b1 = 0.5 * n - 2 / 3 * n2 + 37 / 96 * n3
  const b2 = n2 / 48 + n3 / 15
  const b3 = 17 / 480 * n3
  let xi0 = xi
  let eta0 = eta
  for (let j = 1; j <= 3; j++) {
    const bj = [0, b1, b2, b3][j]
    xi0 -= bj * Math.sin(2 * j * xi) * Math.cosh(2 * j * eta)
    eta0 -= bj * Math.cos(2 * j * xi) * Math.sinh(2 * j * eta)
  }
  const chi = Math.asin(Math.sin(xi0) / Math.cosh(eta0))
  const d1 = 2 * n - 2 / 3 * n2 - 2 * n3
  const d2 = 7 / 3 * n2 - 8 / 5 * n3
  const d3 = 56 / 15 * n3
  let lat = chi + d1 * Math.sin(2 * chi) + d2 * Math.sin(4 * chi) + d3 * Math.sin(6 * chi)
  let lon = Math.atan2(Math.sinh(eta0), Math.cos(xi0))
  lat = lat * 180 / Math.PI
  lon = lon * 180 / Math.PI + (zone * 6 - 183)
  return [lat, lon]
}

export function groundToLatLon(gx, gy, originLon, originLat) {
  if (!originLon && !originLat) return null
  const [oE, oN, zone, isNorth] = latLonToUtm(originLat, originLon)
  return utmToLatLon(gx + oE, gy + oN, zone, isNorth)
}

export function latLonToLocal(lon, lat, originLon, originLat) {
  const [oE, oN] = latLonToUtm(originLat, originLon)
  const [E, N] = latLonToUtm(lat, lon)
  return [E - oE, N - oN]
}
