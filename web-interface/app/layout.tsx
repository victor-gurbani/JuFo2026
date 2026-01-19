import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "JuFo 2026 - Musical Analysis Explorer",
  description: "Interactive visualization and analysis of musical pieces in feature space",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
