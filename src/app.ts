import './style.css'

export function renderApp() {
  document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
    <div class="min-h-screen flex items-center justify-center bg-gray-900 text-white">
      <div class="text-center">
        <h1 class="text-5xl font-bold mb-8">Hello World</h1>
        <p class="text-xl text-gray-300">Welcome to the app!</p>
      </div>
    </div>
  `
}
