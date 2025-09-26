/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'marking': {
          'hover': 'rgba(239, 68, 68, 0.2)',
          'active': 'rgba(239, 68, 68, 0.4)',
          'confirmed': 'rgba(239, 68, 68, 0.6)',
        }
      },
      animation: {
        'pulse-subtle': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}