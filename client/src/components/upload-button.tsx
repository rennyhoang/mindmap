import { useContext, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { SessionContext, TranscriptContext } from "./session-context";

function UploadButton() {
  const sessionContext = useContext(SessionContext);
  const transcriptContext = useContext(TranscriptContext);
  if (!sessionContext || !transcriptContext) {
    throw new Error("ChildComponent must be used within a SessionProvider");
  }
  const setSessionId = sessionContext.setSessionId;
  const setTranscript = transcriptContext.setTranscript;
  const [file, setFile] = useState<File | null>(null);

  const uploadFile = async () => {
    if (!file) {
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`http://127.0.0.1:8000/uploadfile/`, {
        method: "POST",
        body: formData,
      });
      const json = await response.json();
      if (setTranscript) {
        setTranscript(json.transcription);
      }
      if (setSessionId) {
        setSessionId(json.session_id);
      }
    } catch {
      console.log("catch");
    }
  };

  return (
    <div className="flex flex-row gap-4">
      <Input
        className="bg-background"
        name="audioFile"
        type="file"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
          }
        }}
      />
      <Button onClick={uploadFile}>Submit</Button>
    </div>
  );
}

export default UploadButton;
