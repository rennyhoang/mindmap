import { useContext, useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import RecordRTC from "recordrtc";
import { SessionContext, TranscriptContext } from "./session-context";

function RecordButton() {
  const sessionContext = useContext(SessionContext);
  const transcriptContext = useContext(TranscriptContext);
  if (!sessionContext || !transcriptContext) {
    throw new Error("ChildComponent must be used within a SessionProvider");
  }
  const { sessionId, setSessionId } = sessionContext;
  const setTranscript = transcriptContext.setTranscript;

  const [recording, setRecording] = useState(false);
  const socketRef = useRef<WebSocket>(null);
  const recorderRef = useRef<RecordRTC>(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      socketRef.current = new WebSocket("ws://127.0.0.1:8000/transcribe");

      socketRef.current.onopen = () => {
        console.log("WebSocket connection established");
        if (sessionId && socketRef.current) {
          socketRef.current.send(sessionId);
        }
      };

      socketRef.current.onmessage = (e) => {
        if (e.data.startsWith("Session ID")) {
          if (socketRef.current) {
            socketRef.current.close();
          }
          if (setSessionId) {
            setSessionId(e.data.split(": ")[1]);
          }
        } else {
          if (setTranscript) {
            setTranscript(e.data);
          }
        }
      };

      recorderRef.current = new RecordRTC(stream, {
        type: "audio",
        mimeType: "audio/wav",
        recorderType: RecordRTC.StereoAudioRecorder,
        timeSlice: 5000,
        desiredSampRate: 16000,
        numberOfAudioChannels: 1,
        ondataavailable: (blob) => {
          if (
            socketRef.current &&
            socketRef.current.readyState === WebSocket.OPEN
          ) {
            socketRef.current.send(blob);
          }
        },
      });

      setRecording(true);
      recorderRef.current.startRecording();
    } catch (err) {
      console.error("Error accessing microphone:", err);
    }
  };

  const stopRecording = () => {
    if (recorderRef.current) {
      recorderRef.current.stopRecording();
    }
    if (socketRef.current) {
      socketRef.current.send("STOP");
    }
    setRecording(false);
  };

  return (
    <Button
      onClick={recording ? stopRecording : startRecording}
      className="drop-shadow-xl/25"
    >
      {recording ? "Stop Recording" : "Start Recording"}
    </Button>
  );
}

export default RecordButton;
