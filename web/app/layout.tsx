import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Viewport } from "@/components/viewport";
import { SoundEffects } from "@/components/sfx";
import { ClickToEnter } from "@/components/click-to-enter";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DroneStalker - Real-time Drone Tracking & Prediction",
  description:
    "A real-time drone tracking and prediction system built at Junction 2025. Uses machine learning to track drones in video feeds and predict their future trajectories using the FRED dataset.",
  keywords: [
    "drone tracking",
    "machine learning",
    "computer vision",
    "trajectory prediction",
    "FRED dataset",
    "sensor fusion",
    "YOLOv12",
    "event cameras",
  ],
  authors: [
    { name: "Cody Adam" },
    { name: "Jordan Legg" },
    { name: "Jacob Kenney" },
    { name: "Mikus Sturmaniss" },
  ],
  openGraph: {
    title: "DroneStalker - Real-time Drone Tracking & Prediction",
    description:
      "A real-time drone tracking and prediction system built at Junction 2025. Uses machine learning to track drones in video feeds and predict their future trajectories.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "DroneStalker - Real-time Drone Tracking & Prediction",
    description:
      "A real-time drone tracking and prediction system built at Junction 2025.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full w-full">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased h-full w-full crt`}
      >
        <ClickToEnter>
          <SoundEffects />
          <Viewport>{children}</Viewport>
        </ClickToEnter>
      </body>
    </html>
  );
}
