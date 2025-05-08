import { useRef, useContext, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
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
  const [loading, setLoading] = useState<boolean>(false);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const uploadFile = async () => {
    if (!file) {
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`http://213.180.0.37:47947/uploadfile/`, {
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

    setLoading(false);
  };

  return (
    <div className="flex flex-row gap-4">
      <Input
        className="bg-background drop-shadow-xl/25"
        name="audioFile"
        type="file"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
          }
        }}
        disabled={loading ? true : false}
      />
      <Button
        ref={buttonRef}
        onClick={uploadFile}
        className="drop-shadow-xl/25"
        disabled={loading ? true : false}
      >
        {loading ? <Loader2 className="animate-spin"></Loader2> : <></>}
        {!loading ? "Upload" : "Loading"}
      </Button>
    </div>
  );
}

export default UploadButton;
