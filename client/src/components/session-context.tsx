import { createContext, useState, Dispatch, SetStateAction } from 'react';

interface SessionContextType {
    sessionId: string;
    setSessionId: Dispatch<SetStateAction<string>> | null;
}

interface TranscriptType {
    transcript: string;
    setTranscript: Dispatch<SetStateAction<string>> | null;
}
  
export const SessionContext = createContext<SessionContextType>({sessionId: "", setSessionId: null});
export const TranscriptContext = createContext<TranscriptType>({transcript: "", setTranscript: null});

export const SessionProvider = ({ children }: { children: React.ReactNode }) => {
  const [sessionId, setSessionId] = useState<string>("");
  const [transcript, setTranscript] = useState<string>("");

  return (
    <SessionContext.Provider value={{ sessionId, setSessionId }}>
      <TranscriptContext.Provider value={{ transcript, setTranscript }}>
        {children}
      </TranscriptContext.Provider>
    </SessionContext.Provider>
  );
};