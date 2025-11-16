import Link from "next/link";
import { useStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export default function Credits() {
  const dataId = useStore((state) => state.dataId);
  const setDataId = useStore((state) => state.setDataId);
  const reset = useStore((state) => state.reset);

  const missions = ["0", "60", "132", "230"];

  const handleMissionChange = (newDataId: string) => {
    reset();
    // Set the new dataId after reset (reset sets it to "60" by default)
    setDataId(newDataId);
  };

  return (
    <div className="border-4 flex h-full opened flex-col gap-1 p-6 overflow-y-auto text-sm">
      <p>
        This project was built in just 40 hours at Junction 2025: Utopia &
        Dystopia, held in Espoo, Finland.
      </p>
      <p className="mt-2">
        We used the{" "}
        <Link
          href="https://miccunifi.github.io/FRED/"
          className="text-white underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          FRED dataset
        </Link>
        , which contains video frames from both RGB and motion (event) cameras.
      </p>
      <p className="mt-2">
        <b>Main challenges:</b> Clean the dataset, find and track the drone in
        each frame, and train a machine learning model to predict where the
        drone would go next.
      </p>
      <p className="mt-2">
        <b>Side challenge:</b> We also estimated the rotation speed of a fan
        seen in the camera feed using frequency analysis. (see Appendix: Fan
        speed prediction)
      </p>
      <p className="mt-2">
        <b>Vision:</b> Our goal was to showcase incredible data science, AI, and
        ML software through a highly engaging, visual, and gamified experience.
        Rather than presenting technical achievements traditionally, we created
        an interactive narrative that transforms complex machine learning
        concepts into an accessible and exciting journey.
      </p>

      <Title>Mission selector</Title>
      <p>
        You can play again with different dataset entries from the FRED dataset:
      </p>
      <div className="flex flex-wrap gap-2 mb-4 mt-2">
        {missions.map((missionId) => (
          <Button
            key={missionId}
            onClick={() => handleMissionChange(missionId)}
            className={cn(
              "uppercase",
              dataId === missionId && "bg-foreground text-background"
            )}
          >
            Mission {missionId}
          </Button>
        ))}
      </div>

      <Title>The team</Title>

      <p>
        <Link
          href="https://www.linkedin.com/in/codyadam/"
          className="text-white underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          Cody Adam
        </Link>
        : Web experience, Chatbot, Graphic direction, SFX
      </p>
      <p>
        <Link
          href="https://www.linkedin.com/in/404missinglink/"
          className="text-white underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          Jordan Legg
        </Link>
        : Dataset engineering, Data engineering, AI voice
      </p>
      <p>
        <Link
          href="https://www.linkedin.com/in/jacob-kenney1/"
          className="text-white underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          Jacob Kenney
        </Link>
        : ML/AI prediction, Data engineering
      </p>
      <p>
        <Link
          href="https://www.linkedin.com/in/mikussturmanis/"
          className="text-white underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          Mikus Sturmaniss
        </Link>
        : RPM challenge, Object detection
      </p>
      <Title>Challenge and tech stack</Title>
      <p>
        <strong>Challenge:</strong> Sensorfusion - Lights, Camera, Reaction!
      </p>
      <p className="mt-2">
        <strong>Tech challenges:</strong>
      </p>
      <ul className="list-inside ml-4">
        <li>- Gemini API: for the chat with the Commander</li>
        <li>
          - Vultr: multi-hundred gigabytes compute{" "}
          <Link
            href="https://huggingface.co/datasets/takara-ai/FRED-CONVERTED"
            className="text-white underline"
          >
            Dataset
          </Link>
        </li>
        <li>- ElevenLabs: for the voice of the Commander</li>
        <li>- GoDaddy: for domain purchase</li>
      </ul>
      <p className="mt-2">
        <strong>ML/AI stack:</strong>
      </p>
      <ul className="list-inside ml-4">
        <li>- Python</li>
        <li>- PyTorch</li>
        <li>- YOLOv12</li>
        <li>- Custom ViT-T</li>
      </ul>
      <p className="mt-2">
        <strong>Web stack:</strong>
      </p>
      <ul className="list-inside ml-4">
        <li>- Next.js</li>
        <li>- React</li>
        <li>- Zustand</li>
        <li>- Tailwind</li>
        <li>- Vercel</li>
      </ul>
      <p className="mt-2">
        <strong>GitHub repository:</strong>{" "}
        <Link
          href="https://github.com/takara-ai/DroneStalker"
          className="text-white underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          DroneStalker
        </Link>
      </p>
      <Title>Appendix: Fan speed prediction</Title>
      <div className="w-full aspect-video grid grid-cols-2 p-2 gap-2 border-4 border-border/50 relative">
        <div className="relative w-full h-full">
          <video
            src="/data/fan/visualization_fan_varying_rpm_turning_evio_player.mp4"
            autoPlay
            muted
            loop
            className="w-full h-full object-cover"
          />
          <span className="absolute left-1/2 bottom-2 -translate-x-1/2 -translate-y-1/2 bg-background/80 text-white text-xl px-4 py-1 select-none pointer-events-none">
            INPUT
          </span>
        </div>
        <div className="relative w-full h-full">
          <video
            src="/data/fan/visualization_fan_varying_rpm_turning.mp4"
            autoPlay
            muted
            loop
            className="w-full h-full object-contain"
          />
          <span className="absolute left-1/2 bottom-2 -translate-x-1/2 -translate-y-1/2 bg-background/80 text-white text-xl px-4 py-1 select-none pointer-events-none">
            OUTPUT
          </span>
        </div>
      </div>
    </div>
  );
}

function Title({ children }: { children: React.ReactNode }) {
  return <h1 className="text-4xl font-bold not-first:mt-8 mb-4">{children}</h1>;
}
