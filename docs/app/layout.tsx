import { RootProvider } from "fumadocs-ui/provider"
import { Inter } from "next/font/google"
import type { ReactNode } from "react"
import "./global.css"
import type { Metadata } from "next";
import { siteConfig } from "@/public/siteConfig";

const inter = Inter({
  subsets: ["latin"],
})

export const metadata: Metadata = {
	title: {
		default: siteConfig.name,
		template: "%s",
	},
	metadataBase: new URL(siteConfig.url),
	openGraph: {
		url: siteConfig.url,
		title: siteConfig.name,
		siteName: siteConfig.name,
	},
	icons: {
		icon: "/favicon.svg",
	},
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={inter.className}>
      <body>
        <RootProvider>
          {children}
        </RootProvider>
      </body>
    </html>
  )
}
