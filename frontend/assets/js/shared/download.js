const activeUrls = new Set()

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  activeUrls.add(url)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => revokeUrl(url), 0)
}

export function downloadText(text, filename, type = 'text/plain') {
  downloadBlob(new Blob([text], { type }), filename)
}

export function downloadCsv(rows, filename) {
  const csv = rows.map((row) => row.join(',')).join('\n')
  downloadText(csv, filename, 'text/csv')
}

export function revokeUrl(url) {
  if (!url) return
  URL.revokeObjectURL(url)
  activeUrls.delete(url)
}

export function revokeAllUrls() {
  for (const url of activeUrls) {
    URL.revokeObjectURL(url)
  }
  activeUrls.clear()
}
