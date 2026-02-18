import os
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

from werkzeug.security import generate_password_hash

from app import create_app, db
from app.models import DailyCheckIn, FavoriteMeal, MIMChatMessage, Meal, Substance, User, UserProfile


def _complete_profile(user_id: int) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        age=35,
        biological_sex="male",
        time_zone="UTC",
        height_cm=178.0,
        weight_kg=80.0,
        primary_goal="focus consistency",
        fitness_level="moderate",
        diet_style="omnivore",
        medical_conditions="none",
    )


class DataIsolationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_file = Path(tempfile.gettempdir()) / f"coachmim-isolation-{uuid4().hex}.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{cls.db_file.as_posix()}"
        os.environ["SECRET_KEY"] = "test-secret"
        os.environ["ENCRYPTION_REQUIRED"] = "false"

        cls.app = create_app()
        cls.app.config.update(TESTING=True)

        with cls.app.app_context():
            db.drop_all()
            db.create_all()

            user1 = User(
                full_name="User One",
                email="user1@example.com",
                password_hash=generate_password_hash("pass12345"),
            )
            user2 = User(
                full_name="User Two",
                email="user2@example.com",
                password_hash=generate_password_hash("pass12345"),
            )
            db.session.add_all([user1, user2])
            db.session.flush()

            db.session.add_all([_complete_profile(user1.id), _complete_profile(user2.id)])
            db.session.flush()

            db.session.add(
                Meal(
                    user_id=user1.id,
                    eaten_at=datetime(2026, 2, 18, 12, 0),
                    label="Lunch",
                    description="U1_SECRET_MEAL",
                    calories=420,
                )
            )
            meal_u2 = Meal(
                user_id=user2.id,
                eaten_at=datetime(2026, 2, 18, 12, 30),
                label="Lunch",
                description="U2_SECRET_MEAL",
                calories=777,
            )
            db.session.add(meal_u2)

            db.session.add_all(
                [
                    Substance(
                        user_id=user1.id,
                        taken_at=datetime(2026, 2, 18, 8, 0),
                        kind="caffeine",
                        amount="U1_SUBSTANCE_100MG",
                    ),
                    Substance(
                        user_id=user2.id,
                        taken_at=datetime(2026, 2, 18, 9, 0),
                        kind="caffeine",
                        amount="U2_SUBSTANCE_500MG",
                    ),
                ]
            )

            db.session.add_all(
                [
                    DailyCheckIn(
                        user_id=user1.id,
                        day=date(2026, 2, 18),
                        sleep_hours=7.2,
                        productivity=8,
                    ),
                    DailyCheckIn(
                        user_id=user2.id,
                        day=date(1999, 12, 31),
                        sleep_hours=2.0,
                        productivity=1,
                    ),
                ]
            )

            db.session.add_all(
                [
                    FavoriteMeal(user_id=user1.id, name="U1_PRIVATE_FAVORITE"),
                    FavoriteMeal(user_id=user2.id, name="U2_PRIVATE_FAVORITE"),
                ]
            )
            db.session.add_all(
                [
                    MIMChatMessage(user_id=user1.id, role="user", content="U1_CHAT_SECRET"),
                    MIMChatMessage(user_id=user2.id, role="user", content="U2_CHAT_SECRET"),
                ]
            )

            db.session.commit()
            cls.user2_meal_id = meal_u2.id

    @classmethod
    def tearDownClass(cls):
        with cls.app.app_context():
            db.session.remove()
            db.drop_all()
            db.engine.dispose()
        if cls.db_file.exists():
            try:
                cls.db_file.unlink()
            except PermissionError:
                pass

    def setUp(self):
        self.client = self.app.test_client()
        response = self.client.post(
            "/login",
            data={"email": "user1@example.com", "password": "pass12345"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

    def test_user_cannot_open_another_users_meal_edit_page(self):
        response = self.client.get(f"/meal/{self.user2_meal_id}/edit")
        self.assertEqual(response.status_code, 404)

    def test_user_cannot_delete_another_users_meal(self):
        response = self.client.post(
            f"/meal/{self.user2_meal_id}/delete",
            data={"day": "2026-02-18"},
        )
        self.assertEqual(response.status_code, 404)

    def test_timeline_only_shows_current_users_private_content(self):
        response = self.client.get("/timeline")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("U1_SECRET_MEAL", html)
        self.assertNotIn("U2_SECRET_MEAL", html)
        self.assertIn("U1_SUBSTANCE_100MG", html)
        self.assertNotIn("U2_SUBSTANCE_500MG", html)

    def test_checkin_day_manager_does_not_expose_other_users_history_or_favorites(self):
        response = self.client.get("/checkin?day=2026-02-18&view=meal")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("U1_PRIVATE_FAVORITE", html)
        self.assertNotIn("U2_PRIVATE_FAVORITE", html)
        self.assertNotIn("1999-12-31", html)

    def test_ask_mim_history_is_user_scoped(self):
        response = self.client.get("/ask-mim")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("U1_CHAT_SECRET", html)
        self.assertNotIn("U2_CHAT_SECRET", html)


if __name__ == "__main__":
    unittest.main()
