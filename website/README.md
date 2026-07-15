# OmniaBench project page

The project page is a statically exported Next.js site deployed from the repository root by `.github/workflows/pages.yml`.

```bash
npm ci
npm run dev
```

For a local production build that mirrors GitHub Pages:

```bash
NEXT_PUBLIC_BASE_PATH=/OmniaBench npm run build:pages
```
