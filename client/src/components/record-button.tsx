import { useContext, useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import RecordRTC from 'recordrtc';
import { SessionContext, TranscriptContext } from "./session-context";

function RecordButton() {
    const sessionContext = useContext(SessionContext);
    const transcriptContext = useContext(TranscriptContext);
    if (!sessionContext || !transcriptContext) {
        throw new Error(
          "ChildComponent must be used within a SessionProvider"
        );
    }
    const { sessionId, setSessionId } = sessionContext;
    const { transcript, setTranscript } = transcriptContext;

    const [recording, setRecording] = useState(false);
    const socketRef = useRef<WebSocket>(null);
    const recorderRef = useRef<RecordRTC>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
    // Only scroll if recording is active AND the ref is attached
    if (recording && textareaRef.current) {
      const textarea = textareaRef.current;
      textarea.scrollTop = textarea.scrollHeight;
    }
  }, [transcript, recording]); // Dependencies: run effect when these change
    
    const onTextareaChange = async (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        if (setTranscript && e.currentTarget.value) {
            setTranscript(e.currentTarget.value);
        }
    };

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
                    if (setSessionId) {setSessionId(e.data.split(": ")[1])};
                } else {
                    if (textareaRef.current) {textareaRef.current.value = transcript};
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
                    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
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
        <div className="flex flex-col gap-4">
            <Button onClick={recording ? stopRecording : startRecording}>
                {recording ? "Stop Recording" : "Start Recording"}
            </Button>
            <Textarea ref={textareaRef} className="bg-background resize-none" onChange={onTextareaChange}/>
        </div>
    );
}

export default RecordButton;