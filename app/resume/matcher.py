"""Lightweight JD-to-resume matching for resume-aware interview planning."""

from __future__ import annotations

from app.domain.skill_aliases import find_skill_mention
from app.models.jd import SkillMatrix
from app.models.resume import ResumeJDMatch, ResumeProfile


def match_resume_to_jd(profile: ResumeProfile, skill_matrix: SkillMatrix) -> ResumeJDMatch:
    """Match JD skills to resume skills and projects with simple deterministic rules."""
    resume_skill_text = " ".join(profile.skills)
    project_text_by_name = {
        project.name: " ".join(
            [
                project.name,
                " ".join(project.tech_stack),
                project.description,
                " ".join(project.key_contributions),
                " ".join(project.potential_deep_dive_points),
            ]
        )
        for project in profile.projects
    }
    full_resume_text = " ".join([resume_skill_text, *project_text_by_name.values()])

    matched_skills: list[str] = []
    missing_skills: list[str] = []
    match_evidence: dict[str, list[str]] = {}
    project_skill_map: dict[str, list[str]] = {name: [] for name in project_text_by_name}

    for skill in sorted(skill_matrix.skills, key=lambda item: item.weight, reverse=True):
        mention = find_skill_mention(full_resume_text, skill.name)
        if mention:
            matched_skills.append(skill.name)
            match_evidence[skill.name] = [mention]
            for project_name, project_text in project_text_by_name.items():
                if find_skill_mention(project_text, skill.name):
                    project_skill_map[project_name].append(skill.name)
        elif skill.is_required or skill.weight >= 0.6:
            missing_skills.append(skill.name)

    project_skill_map = {name: skills for name, skills in project_skill_map.items() if skills}
    relevant_projects = sorted(
        project_skill_map,
        key=lambda name: len(project_skill_map[name]),
        reverse=True,
    )[:3]

    interview_focus: list[str] = []
    for project_name in relevant_projects:
        skills = ", ".join(project_skill_map[project_name][:4])
        interview_focus.append(f"Probe project '{project_name}' around: {skills}")

    for concern in profile.concerns[:3]:
        interview_focus.append(f"Verify resume concern: {concern}")

    for skill in missing_skills[:3]:
        interview_focus.append(f"Check JD skill not mentioned in resume (not assumed absent): {skill}")

    if not interview_focus and matched_skills:
        interview_focus.append(
            "Validate whether listed skills were used hands-on or only mentioned in passing."
        )

    return ResumeJDMatch(
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        unmentioned_skills=missing_skills,
        relevant_projects=relevant_projects,
        interview_focus=interview_focus,
        project_skill_map=project_skill_map,
        match_evidence=match_evidence,
    )
