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
      <Title>Mission selector</Title>
      <div className="flex flex-wrap gap-2 mb-4">
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
          - Vultr: multi-hundred GB compute{" "}
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
          takara-ai/DroneStalker
        </Link>
      </p>
    </div>
  );
}

function Title({ children }: { children: React.ReactNode }) {
  return <h1 className="text-4xl font-bold not-first:mt-8 mb-4">{children}</h1>;
}
