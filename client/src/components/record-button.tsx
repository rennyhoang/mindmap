import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import RecordRTC from 'recordrtc';

function RecordButton({ setSessionId }: { setSessionId: Function}) {
    const [transcript, setTranscript] = useState("");
    const [recording, setRecording] = useState(false);
    const socketRef = useRef<WebSocket>(null);
    const recorderRef = useRef<RecordRTC>(null);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            socketRef.current = new WebSocket("ws://localhost:8000/transcribe");

            socketRef.current.onopen = () => {
                console.log("WebSocket connection established");
            };

            socketRef.current.onmessage = (e) => {
                if (e.data.startsWith("Session ID")) {
                    if (socketRef.current) {
                        socketRef.current.close();
                    }
                    setSessionId(e.data.split(": ")[1]);
                } else {
                    if (!e.data.startsWith("Error: 'bytes'")) {
                      setTranscript((prev) => prev + " " + e.data);
                    }
                }
            };

            recorderRef.current = new RecordRTC(stream, {
                type: "audio",
                mimeType: "audio/wav", // Send uncompressed audio
                recorderType: RecordRTC.StereoAudioRecorder,
                timeSlice: 5000, // Send chunks every second
                desiredSampRate: 16000, // Recommended for speech recognition
                numberOfAudioChannels: 1, // Mono channel for better compatibility
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
    };

    return (
        <>
            <Button onClick={recording ? stopRecording : startRecording}>
                {recording ? "Stop Recording" : "Start Recording"}
            </Button>
            <Textarea placeholder={transcript} disabled/>
        </>
    );
}

export default RecordButton;