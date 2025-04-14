import { useContext, useState, useEffect } from "react";
import { ReactFlow, MiniMap, Controls, Background } from "@xyflow/react";
import { SessionContext } from "./session-context"

import "@xyflow/react/dist/style.css";

function Flow() {
  const context = useContext(SessionContext);

  if (!context) {
      throw new Error(
        "ChildComponent must be used within a SessionProvider"
      );
  }

  const { sessionId, } = context;


  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);

  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const response = await fetch(
          `http://localhost:8000/graph/${sessionId}`,
        );
        const graphData = await response.json();

        setNodes(graphData.nodes);
        setEdges(graphData.edges);
      } catch (error) {
        console.error("Error fetching graph data:", error);
      }
    };

    fetchGraphData();
  }, [sessionId]);

  return (
    <div className="w-screen h-screen absolute z-0">
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <MiniMap zoomable pannable />
        <Controls />
        <Background />
      </ReactFlow>
    </div>
  );
}

export default Flow;
