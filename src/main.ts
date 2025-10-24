import './style.css'
import { renderApp } from './app.ts'
import { renderDemo } from './demo.ts'

// Simple router
function router() {
  const path = window.location.pathname
  
  if (path === '/demo') {
    renderDemo()
  } else {
    renderApp()
  }
}

// Handle browser back/forward
window.addEventListener('popstate', router)

// Initial route
router()
