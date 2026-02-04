import { NextResponse } from "next/server";
import { getJob } from "@/lib/cloudJobs";

export const runtime = "nodejs";

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const job = getJob(id);
  if (!job) {
    return NextResponse.json({ error: `Unknown job: ${id}` }, { status: 404 });
  }
  return NextResponse.json({ job });
}
