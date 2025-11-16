# DroneStalker

A real-time drone tracking and prediction system built in 40 hours at **Junction 2025: Utopia & Dystopia**, held in Espoo, Finland.

## Overview

DroneStalker is an interactive web application that tracks drones in video feeds and uses machine learning to predict their future trajectories. The project uses the [FRED dataset](https://miccunifi.github.io/FRED/), which contains video frames from both RGB and motion (event) cameras.

## Vision & Goal

The primary goal of DroneStalker was to showcase incredible data science, AI, and ML software through a highly engaging and visual experience. Rather than presenting the technical achievements in a traditional format, we created a **gamified scenario** that transforms complex machine learning concepts into an interactive, narrative-driven experience.

### The Gamified Experience

Users are immersed in a military defense simulation where they play the role of a hacker working under a trash-talking Commander. The experience unfolds through progressive stages:

1. **Mission Introduction**: Users activate camera feeds and attempt manual targeting
2. **Code Development**: Users write actual code to implement motion detection algorithms
3. **Feature Unlocking**: As code is completed, new capabilities unlock (motion detection, tracking, position prediction)
4. **Interactive Combat**: Users engage with the drone tracking system in real-time, seeing their code in action
5. **AI-Powered Guidance**: A Commander AI (powered by Gemini) provides context-aware, humorous guidance throughout the journey

This approach makes advanced ML/AI concepts—like computer vision, object tracking, and trajectory prediction—accessible and exciting. Users don't just read about the technology; they experience it firsthand through an interactive, visual, and gamified narrative that demonstrates the power of modern data science and machine learning.

## Main Challenges

- **Dataset Cleaning**: Process and clean the FRED dataset for training
- **Drone Detection & Tracking**: Find and track the drone in each frame
- **Trajectory Prediction**: Train a machine learning model to predict where the drone will go next

## Side Challenge

**Fan Speed Prediction**: Estimate the rotation speed of a fan seen in the camera feed using frequency analysis. See the visualization in the credits section of the application.

## Challenge

**Sensorfusion - Lights, Camera, Reaction!**

## Tech Stack

### ML/AI Stack

- Python
- PyTorch
- YOLOv12
- Custom ViT-T

### Web Stack

- Next.js
- React
- Zustand
- Tailwind CSS
- Vercel

### Services & Infrastructure

- **Gemini API**: Chat with the Commander
- **Vultr**: Multi-hundred gigabytes compute for dataset processing
- **ElevenLabs**: Voice synthesis for the Commander
- **GoDaddy**: Domain hosting

## Dataset

The project uses the [FRED dataset](https://miccunifi.github.io/FRED/), which is also available on [Hugging Face](https://huggingface.co/datasets/takara-ai/FRED-CONVERTED).

## Getting Started

### Prerequisites

- Node.js 18+
- npm, yarn, pnpm, or bun

### Installation

```bash
# Clone the repository
git clone https://github.com/takara-ai/DroneStalker.git
cd DroneStalker

# Install dependencies
npm install
# or
yarn install
# or
pnpm install
```

### Environment Variables

Create a `.env` file in the `/web` directory with the following API keys:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

**Note**: You'll need to obtain these API keys from their respective services:

- **OpenRouter**: For API access
- **ElevenLabs**: For voice synthesis (both `ELEVENLABS_VOICE_ID` and `ELEVENLABS_API_KEY` are required)

### Development

Run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Mission Selector

The application includes a mission selector that allows you to play with different dataset entries from the FRED dataset:

- Mission 0
- Mission 60
- Mission 132
- Mission 230

## The Team

- **[Cody Adam](https://www.linkedin.com/in/codyadam/)**: Web experience, Chatbot, Graphic direction, SFX
- **[Jordan Legg](https://www.linkedin.com/in/404missinglink/)**: Dataset engineering, Data engineering, AI voice
- **[Jacob Kenney](https://www.linkedin.com/in/jacob-kenney1/)**: ML/AI prediction, Data engineering
- **[Mikus Sturmaniss](https://www.linkedin.com/in/mikussturmanis/)**: RPM challenge, Object detection

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Junction 2025: Utopia & Dystopia hackathon organizers
- FRED dataset creators and contributors
- All the open-source libraries and tools that made this project possible
