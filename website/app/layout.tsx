import type { Metadata } from "next";
import "./globals.css";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const publicOrigin = `https://scuuy.github.io${basePath}`;

export const metadata: Metadata = {
  metadataBase: new URL(`https://scuuy.github.io${basePath || "/"}`),
  title: "OmniaBench — Benchmarking General AI Agents",
  description: "A broad and diagnostic benchmark for evaluating general AI agents across 354 real-world scenario domains.",
  icons: { icon: `${basePath}/favicon.png`, shortcut: `${basePath}/favicon.png` },
  openGraph: {
    title: "OmniaBench — Benchmarking General AI Agents",
    description: "90 domains. 1,431 tasks. A diagnostic benchmark for general AI agents.",
    images: [{ url: `${publicOrigin}/og.png`, width: 1536, height: 1024, alt: "OmniaBench benchmark overview" }],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "OmniaBench — Benchmarking General AI Agents",
    description: "90 domains. 1,431 tasks. A diagnostic benchmark for general AI agents.",
    images: [`${publicOrigin}/og.png`],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
