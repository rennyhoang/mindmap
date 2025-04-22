import { useContext } from "react";
import { Textarea } from "@/components/ui/textarea";
import { TranscriptContext } from "./session-context";

function Transcript() {
  const transcriptContext = useContext(TranscriptContext);
  const transcript = transcriptContext.transcript;

  return (
      <Textarea
        className="bg-background resize-none text-xs drop-shadow-xl/25"
        value={transcript}
        readOnly
      />
  );
}

export default Transcript;
