export const $ = (id) => document.getElementById(id)

export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

export function fmtTime(seconds) {
  if (!isFinite(seconds)) return '0:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s < 10 ? '0' : ''}${s}`
}
