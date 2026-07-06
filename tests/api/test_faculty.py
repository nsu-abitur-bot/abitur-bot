"""Тесты API справочника факультетов и направлений (TestClient + фейк-сервис)."""

from dataclasses import dataclass, field
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.faculty import get_faculty_service
from db.postgres.models import EDUCATION_LEVELS, EducationLevel


@dataclass
class FakeFaculty:
    id: str
    name: str
    aliases: list = field(default_factory=list)
    is_active: bool = True


@dataclass
class FakeProgram:
    id: str
    faculty_id: str
    name: str
    level: str
    code: Optional[str] = None
    is_active: bool = True


class FakeFacultyService:
    """Минимальная имитация FacultyService в памяти."""

    def __init__(self):
        self.faculties: dict[str, FakeFaculty] = {}
        self.programs: dict[str, FakeProgram] = {}
        self._seq = 0

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    async def get_all_faculties(self, only_active: bool = True):
        items = list(self.faculties.values())
        if only_active:
            items = [f for f in items if f.is_active]
        return sorted(items, key=lambda f: f.name)

    async def get_faculty_by_id(self, faculty_id: str):
        return self.faculties.get(faculty_id)

    async def create_faculty(self, name, aliases=None, is_active=True):
        faculty = FakeFaculty(
            id=self._next_id("faculty"),
            name=name.strip(),
            aliases=[a.strip() for a in (aliases or []) if a and a.strip()],
            is_active=is_active,
        )
        self.faculties[faculty.id] = faculty
        return faculty

    async def update_faculty(self, faculty_id, name=None, aliases=None, is_active=None):
        faculty = self.faculties.get(faculty_id)
        if faculty is None:
            return None
        if name is not None:
            faculty.name = name.strip()
        if aliases is not None:
            faculty.aliases = [a.strip() for a in aliases if a and a.strip()]
        if is_active is not None:
            faculty.is_active = is_active
        return faculty

    async def delete_faculty(self, faculty_id):
        if faculty_id not in self.faculties:
            return False
        del self.faculties[faculty_id]
        self.programs = {
            pid: p for pid, p in self.programs.items() if p.faculty_id != faculty_id
        }
        return True

    async def get_programs_by_faculty(self, faculty_id, level=None, only_active=True):
        items = [p for p in self.programs.values() if p.faculty_id == faculty_id]
        if only_active:
            items = [p for p in items if p.is_active]
        if level is not None:
            items = [p for p in items if p.level == level]
        return sorted(items, key=lambda p: p.name)

    async def get_program_by_id(self, program_id):
        return self.programs.get(program_id)

    async def create_program(
        self,
        faculty_id,
        name,
        level=EducationLevel.bachelor.value,
        code=None,
        is_active=True,
    ):
        if level not in EDUCATION_LEVELS:
            raise ValueError(f"Недопустимый уровень образования: '{level}'")
        program = FakeProgram(
            id=self._next_id("program"),
            faculty_id=faculty_id,
            name=name.strip(),
            level=level,
            code=code.strip() if code else None,
            is_active=is_active,
        )
        self.programs[program.id] = program
        return program

    async def update_program(
        self, program_id, name=None, level=None, code=None, is_active=None
    ):
        program = self.programs.get(program_id)
        if program is None:
            return None
        if name is not None:
            program.name = name.strip()
        if level is not None:
            if level not in EDUCATION_LEVELS:
                raise ValueError(f"Недопустимый уровень образования: '{level}'")
            program.level = level
        if code is not None:
            program.code = code.strip() or None
        if is_active is not None:
            program.is_active = is_active
        return program

    async def delete_program(self, program_id):
        if program_id not in self.programs:
            return False
        del self.programs[program_id]
        return True


@pytest.fixture
def service() -> FakeFacultyService:
    return FakeFacultyService()


@pytest.fixture
def client(service: FakeFacultyService):
    app.dependency_overrides[get_faculty_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_faculty_service, None)


def _create_faculty(client: TestClient, name="Факультет информационных технологий"):
    return client.post(
        "/api/v1/faculties",
        json={"name": name, "aliases": ["ФИТ"], "is_active": True},
    )


def test_create_faculty(client: TestClient):
    response = _create_faculty(client)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Факультет информационных технологий"
    assert data["aliases"] == ["ФИТ"]
    assert data["is_active"] is True
    assert data["programs"] == []


def test_create_program(client: TestClient):
    faculty_id = _create_faculty(client).json()["id"]

    response = client.post(
        f"/api/v1/faculties/{faculty_id}/programs",
        json={"name": "Программная инженерия", "code": "09.03.04", "level": "bachelor"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Программная инженерия"
    assert data["code"] == "09.03.04"
    assert data["level"] == "bachelor"
    assert data["faculty_id"] == faculty_id


def test_list_faculties_with_programs(client: TestClient):
    faculty_id = _create_faculty(client).json()["id"]
    client.post(
        f"/api/v1/faculties/{faculty_id}/programs",
        json={"name": "Программная инженерия", "level": "master"},
    )

    response = client.get("/api/v1/faculties")

    assert response.status_code == 200
    faculties = response.json()
    assert len(faculties) == 1
    assert len(faculties[0]["programs"]) == 1
    assert faculties[0]["programs"][0]["level"] == "master"


def test_list_includes_inactive_faculties(client: TestClient):
    client.post(
        "/api/v1/faculties",
        json={"name": "Выключенный факультет", "aliases": [], "is_active": False},
    )

    response = client.get("/api/v1/faculties")

    assert response.status_code == 200
    names = {f["name"] for f in response.json()}
    assert "Выключенный факультет" in names


def test_update_faculty(client: TestClient):
    faculty_id = _create_faculty(client).json()["id"]

    response = client.put(
        f"/api/v1/faculties/{faculty_id}",
        json={"name": "Новое имя", "is_active": False},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Новое имя"
    assert data["is_active"] is False


def test_update_faculty_not_found(client: TestClient):
    response = client.put("/api/v1/faculties/missing", json={"name": "X"})
    assert response.status_code == 404


def test_update_program(client: TestClient):
    faculty_id = _create_faculty(client).json()["id"]
    program_id = client.post(
        f"/api/v1/faculties/{faculty_id}/programs",
        json={"name": "Программная инженерия", "level": "bachelor"},
    ).json()["id"]

    response = client.put(
        f"/api/v1/faculties/programs/{program_id}",
        json={"level": "master", "code": "09.04.04"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["level"] == "master"
    assert data["code"] == "09.04.04"


def test_delete_program(client: TestClient):
    faculty_id = _create_faculty(client).json()["id"]
    program_id = client.post(
        f"/api/v1/faculties/{faculty_id}/programs",
        json={"name": "Программная инженерия", "level": "bachelor"},
    ).json()["id"]

    response = client.delete(f"/api/v1/faculties/programs/{program_id}")
    assert response.status_code == 204

    assert client.get("/api/v1/faculties").json()[0]["programs"] == []


def test_delete_faculty(client: TestClient):
    faculty_id = _create_faculty(client).json()["id"]

    response = client.delete(f"/api/v1/faculties/{faculty_id}")
    assert response.status_code == 204
    assert client.get("/api/v1/faculties").json() == []


def test_delete_faculty_not_found(client: TestClient):
    response = client.delete("/api/v1/faculties/missing")
    assert response.status_code == 404


def test_create_program_invalid_level(client: TestClient):
    faculty_id = _create_faculty(client).json()["id"]

    response = client.post(
        f"/api/v1/faculties/{faculty_id}/programs",
        json={"name": "Кибербезопасность", "level": "phd"},
    )

    assert response.status_code == 400


def test_create_program_faculty_not_found(client: TestClient):
    response = client.post(
        "/api/v1/faculties/missing/programs",
        json={"name": "Программная инженерия", "level": "bachelor"},
    )
    assert response.status_code == 404
