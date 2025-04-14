import { createContext, useState, Dispatch, SetStateAction } from 'react';

interface SessionContextType {
    sessionId: string | null;
    setSessionId: Dispatch<SetStateAction<string | null>>;
  }
  
export const SessionContext = createContext<SessionContextType | undefined>(
    undefined
);

export const SessionProvider = ({ children }: { children: React.ReactNode }) => {
  const [sessionId, setSessionId] = useState<string | null>("");

  return (
    <SessionContext.Provider value={{ sessionId, setSessionId }}>
      {children}
    </SessionContext.Provider>
  );
};