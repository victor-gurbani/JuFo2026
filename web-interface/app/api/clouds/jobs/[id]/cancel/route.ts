import { NextResponse } from "next/server";
import { cancelJob, getJob } from "@/lib/cloudJobs";

export const runtime = "nodejs";

export async function POST(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const job = getJob(id);
  if (!job) {
    return NextResponse.json({ error: `Unknown job: ${id}` }, { status: 404 });
  }

  const result = cancelJob(id);
  if (!result.ok) {
    return NextResponse.json({ error: result.error }, { status: 400 });
  }
  return NextResponse.json({ ok: true });
}
