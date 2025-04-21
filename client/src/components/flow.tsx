import { useCallback, useContext, useState, useEffect } from "react";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  applyNodeChanges,
  applyEdgeChanges,
  NodeChange,
  EdgeChange,
  Panel,
} from "@xyflow/react";
import { SessionContext, TranscriptContext } from "./session-context";

import "@xyflow/react/dist/style.css";

function Flow() {
  const sessionContext = useContext(SessionContext);
  const transcriptContext = useContext(TranscriptContext);
  if (!sessionContext || !transcriptContext) {
    throw new Error("ChildComponent must be used within a SessionProvider");
  }
  const { sessionId, setSessionId } = sessionContext;
  const { transcript } = transcriptContext;

  const [title, setTitle] = useState("Untitled");
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);

  const onNodesChange = useCallback(
    (changes: NodeChange<never>[]) =>
      setNodes((nds) => applyNodeChanges(changes, nds)),
    [],
  );
  const onEdgesChange = useCallback(
    (changes: EdgeChange<never>[]) =>
      setEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  );

  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/graph/`, {
          method: "POST",
          headers: {
            "Content-type": "application/json",
          },
          body: JSON.stringify({
            transcript: transcript,
            sessionId: sessionId,
          }),
        });
        const graphData = await response.json();

        if (setSessionId) {
          setSessionId(graphData.session_id);
        }
        setNodes(graphData.nodes);
        setEdges(graphData.edges);
        setTitle(graphData.title);
      } catch (error) {
        console.error("Error fetching graph data:", error);
      }
    };

    fetchGraphData();
  }, [sessionId]);

  return (
    <div className="w-screen h-screen absolute z-0">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
      >
        <h1 className="m-4 text-2xl">{title}</h1>
        <MiniMap zoomable pannable />
        <Controls />
        <Background />
      </ReactFlow>
    </div>
  );
}

export default Flow;
