import React, { useState, useContext } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Loader2 } from "lucide-react";
import { SessionContext } from "./session-context";

interface QaResponse {
  answer: string;
}

export default function QaForm() {
  const sessionContext = useContext(SessionContext);
  if (!sessionContext) {
    throw new Error("ChildComponent must be used within a SessionProvider");
  }
  const sessionId = sessionContext.sessionId;

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setAnswer("");

    try {
      const res = await fetch("https://213.180.0.37:47947/qa/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: sessionId, question: question }),
      });
      if (!res.ok) {
        throw new Error(`Status ${res.status}`);
      }
      const data: QaResponse = await res.json();
      setAnswer(data.answer);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="drop-shadow-xl/25">
      <CardHeader>
        <CardTitle>Ask a Question</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid w-full items-center gap-1.5">
            <Label htmlFor="question">Question</Label>
            <Input
              id="question"
              placeholder="Type your question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={loading}
            />
          </div>
          <Button type="submit" disabled={loading || !question || !sessionId}>
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Ask
          </Button>
        </form>

        {answer && (
          <div className="mt-6">
            <Label>Answer</Label>
            <Textarea readOnly value={answer} />
          </div>
        )}

        {error && <p className="mt-2 text-sm text-red-600">Error: {error}</p>}
      </CardContent>
    </Card>
  );
}
