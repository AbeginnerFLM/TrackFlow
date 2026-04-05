export function createLogger(bodyId, emptyId) {
  const body = document.getElementById(bodyId)
  const emptyIdValue = emptyId

  function log(message, type = '') {
    const empty = document.getElementById(emptyIdValue)
    if (empty) empty.remove()
    const entry = document.createElement('div')
    entry.className = 'le' + (type ? ' ' + type : '')
    const t = new Date()
    const ts = `${t.getHours()}:${String(t.getMinutes()).padStart(2, '0')}:${String(t.getSeconds()).padStart(2, '0')}`
    entry.innerHTML = `<span class="ts">${ts}</span><span class="m">${message}</span>`
    body.appendChild(entry)
    body.scrollTop = body.scrollHeight
  }

  function clear() {
    body.innerHTML = `<div class="log-empty" id="${emptyIdValue}">Ready</div>`
  }

  return { log, clear }
}
