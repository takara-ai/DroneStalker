import Link from "next/link";

export default function Credits() {
  return (
    <div className="border-4 flex h-full opened flex-col gap-1 p-6 overflow-y-auto text-sm">
      <h1 className="text-2xl font-bold">The Team</h1>
      <p>Cody Adam: Web experience</p>
      <p>Jordan Legg:</p>
      <p>Jacob Kenney:</p>
      <p>Mikus Sturmaniss:</p>
      <h1 className="text-2xl font-bold mt-4">Challenge and tech stack</h1>
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
    </div>
  );
}
