/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                glasc: {
                    bg: "#0a0c10",
                    panel: "rgba(28, 35, 49, 0.4)",
                    neon: "#00eeff",
                    warning: "#ff9900",
                    success: "#00ff88"
                }
            },
            fontFamily: {
                mono: ['"Fira Code"', 'monospace'],
                sans: ['"Inter"', 'sans-serif']
            }
        },
    },
    plugins: [],
}
